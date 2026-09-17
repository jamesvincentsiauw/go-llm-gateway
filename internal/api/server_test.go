package api

import (
	"bufio"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/redis/go-redis/v9"

	"github.com/jamesvincentsiauw/go-llm-gateway/internal/breaker"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/failover"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/fakellm"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/metrics"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/provider"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/quota"
)

const testKey = "gw-test-key"

type env struct {
	gateway *httptest.Server
	redis   *miniredis.Miniredis
}

func newEnv(t *testing.T, limit int) *env {
	t.Helper()
	upstream := httptest.NewServer(fakellm.Handler())
	t.Cleanup(upstream.Close)
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { _ = rdb.Close() })

	primary := provider.NewOpenAI("primary", upstream.URL+"/v1", "", nil)
	backup := provider.NewOpenAI("backup", upstream.URL+"/v1", "", nil)
	tgt := func(p provider.Provider, model string) failover.Target {
		return failover.Target{Provider: p, Model: model, Breaker: breaker.New(100, time.Minute)}
	}
	m := metrics.New(prometheus.NewRegistry())
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	srv := &Server{
		Routes: map[string][]failover.Target{
			"steady": {tgt(primary, "fake")},
			"flaky":  {tgt(primary, "fake-cut-6"), tgt(backup, "fake")},
			"broken": {tgt(primary, "fake-status-503"), tgt(backup, "fake-status-500")},
			"picky":  {tgt(primary, "fake-status-400"), tgt(backup, "fake")},
		},
		Clients:          map[[32]byte]Client{HashKey(testKey): {Name: "tester", DailyTokenLimit: limit}},
		Engine:           &failover.Engine{Observer: m, Log: log},
		Quota:            quota.New(rdb),
		DefaultMaxTokens: 200,
		MaxTokensLimit:   1000,
		Metrics:          m,
		Log:              log,
	}
	gw := httptest.NewServer(srv.Handler())
	t.Cleanup(gw.Close)
	return &env{gateway: gw, redis: mr}
}

func (e *env) post(t *testing.T, key, body string) *http.Response {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, e.gateway.URL+"/v1/chat/completions", strings.NewReader(body))
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = resp.Body.Close() })
	return resp
}

func chatBody(model string, stream bool) string {
	return `{"model":"` + model + `","stream":` + strconv.FormatBool(stream) +
		`,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":[{"type":"text","text":"tell me"}]}]}`
}

func decode(t *testing.T, r io.Reader) map[string]any {
	t.Helper()
	var v map[string]any
	if err := json.NewDecoder(r).Decode(&v); err != nil {
		t.Fatal(err)
	}
	return v
}

func errorCode(t *testing.T, resp *http.Response) string {
	t.Helper()
	body := decode(t, resp.Body)
	e, _ := body["error"].(map[string]any)
	code, _ := e["code"].(string)
	if code == "" {
		code, _ = e["type"].(string)
	}
	return code
}

func TestRejectsBadRequests(t *testing.T) {
	e := newEnv(t, 0)
	tests := []struct {
		name, key, body string
		status          int
		code            string
	}{
		{"no key", "", chatBody("steady", false), 401, "invalid_api_key"},
		{"wrong key", "nope", chatBody("steady", false), 401, "invalid_api_key"},
		{"unknown model", testKey, chatBody("gpt-9", false), 404, "model_not_found"},
		{"bad json", testKey, `{"model":`, 400, "invalid_request_error"},
		{"no messages", testKey, `{"model":"steady","messages":[]}`, 400, "invalid_request_error"},
		{"tool role", testKey, `{"model":"steady","messages":[{"role":"tool","content":"x"}]}`, 400, "invalid_request_error"},
		{"image part", testKey, `{"model":"steady","messages":[{"role":"user","content":[{"type":"image_url"}]}]}`, 400, "invalid_request_error"},
		{"max tokens too high", testKey, `{"model":"steady","max_tokens":5000,"messages":[{"role":"user","content":"x"}]}`, 400, "invalid_request_error"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := e.post(t, tt.key, tt.body)
			if resp.StatusCode != tt.status {
				t.Fatalf("status = %d, want %d", resp.StatusCode, tt.status)
			}
			if code := errorCode(t, resp); code != tt.code {
				t.Fatalf("code = %q, want %q", code, tt.code)
			}
		})
	}
}

func TestNonStreamingCompletion(t *testing.T) {
	e := newEnv(t, 10_000)
	resp := e.post(t, testKey, chatBody("flaky", false))
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if resp.Header.Get("X-Gateway-Upstreams") != "primary,backup" {
		t.Fatalf("upstreams header = %q", resp.Header.Get("X-Gateway-Upstreams"))
	}
	body := decode(t, resp.Body)
	choice := body["choices"].([]any)[0].(map[string]any)
	if got := choice["message"].(map[string]any)["content"]; got != fakellm.Passage {
		t.Fatalf("content = %q", got)
	}
	if choice["finish_reason"] != "stop" || body["object"] != "chat.completion" {
		t.Fatalf("body = %v", body)
	}
	if fo := body["x_gateway"].(map[string]any)["failovers"].([]any); len(fo) != 1 || fo[0] != "primary->backup" {
		t.Fatalf("failovers = %v", fo)
	}
}

type sse struct {
	text     string
	comments []string
	finish   string
	usage    map[string]any
	done     bool
	errors   []map[string]any
}

func readSSE(t *testing.T, r io.Reader) sse {
	t.Helper()
	var out sse
	var sb strings.Builder
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		line := sc.Text()
		if c, ok := strings.CutPrefix(line, ": "); ok {
			out.comments = append(out.comments, c)
			continue
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		if data == "[DONE]" {
			out.done = true
			continue
		}
		var ev map[string]any
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			t.Fatal(err)
		}
		if e, ok := ev["error"].(map[string]any); ok {
			out.errors = append(out.errors, e)
		}
		if u, ok := ev["usage"].(map[string]any); ok {
			out.usage = u
		}
		for _, c := range ev["choices"].([]any) {
			choice := c.(map[string]any)
			if s, ok := choice["delta"].(map[string]any)["content"].(string); ok {
				sb.WriteString(s)
			}
			if f, ok := choice["finish_reason"].(string); ok {
				out.finish = f
			}
		}
	}
	out.text = sb.String()
	return out
}

func TestStreamingSurvivesMidStreamFailover(t *testing.T) {
	e := newEnv(t, 10_000)
	resp := e.post(t, testKey, chatBody("flaky", true))
	if resp.StatusCode != 200 || resp.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status = %d, content type = %q", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	if resp.Header.Get("X-Gateway-Quota-Remaining") == "" {
		t.Fatal("missing quota header")
	}

	got := readSSE(t, resp.Body)
	if got.text != fakellm.Passage {
		t.Fatalf("streamed text = %q", got.text)
	}
	if len(got.comments) != 1 || got.comments[0] != "gateway failover from=primary to=backup" {
		t.Fatalf("comments = %v", got.comments)
	}
	if got.finish != "stop" || !got.done || got.usage == nil {
		t.Fatalf("finish = %q, done = %v, usage = %v", got.finish, got.done, got.usage)
	}
}

func TestAllUpstreamsDownReturns503BeforeStreaming(t *testing.T) {
	e := newEnv(t, 0)
	resp := e.post(t, testKey, chatBody("broken", true))
	if resp.StatusCode != 503 || !strings.HasPrefix(resp.Header.Get("Content-Type"), "application/json") {
		t.Fatalf("status = %d, content type = %q", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	if code := errorCode(t, resp); code != "upstream_unavailable" {
		t.Fatalf("code = %q", code)
	}
}

func TestUpstreamRejectionIsNotRetried(t *testing.T) {
	e := newEnv(t, 0)
	resp := e.post(t, testKey, chatBody("picky", false))
	if resp.StatusCode != 400 {
		t.Fatalf("status = %d, want 400 passed through instead of trying the backup", resp.StatusCode)
	}
}

func TestQuotaIsReservedThenSettled(t *testing.T) {
	e := newEnv(t, 1000)
	day := "quota:tester:" + time.Now().UTC().Format("2006-01-02")

	// Reservation is input estimate plus max_tokens, so 900 fits once but not twice.
	body := `{"model":"steady","max_tokens":900,"messages":[{"role":"user","content":"tell me"}]}`
	if resp := e.post(t, testKey, body); resp.StatusCode != 200 {
		t.Fatalf("first request status = %d", resp.StatusCode)
	}
	used, _ := e.redis.Get(day)
	passageTokens := len(strings.Fields(fakellm.Passage))
	if used != strconv.Itoa(2+passageTokens) {
		t.Fatalf("settled usage = %s, want %d (reported by the upstream)", used, 2+passageTokens)
	}

	body = `{"model":"steady","max_tokens":990,"messages":[{"role":"user","content":"tell me"}]}`
	resp := e.post(t, testKey, body)
	if resp.StatusCode != 429 {
		t.Fatalf("status = %d, want 429", resp.StatusCode)
	}
	if code := errorCode(t, resp); code != "daily_token_limit" {
		t.Fatalf("code = %q", code)
	}
}

func TestModelsListsRoutes(t *testing.T) {
	e := newEnv(t, 0)
	req, _ := http.NewRequest(http.MethodGet, e.gateway.URL+"/v1/models", nil)
	req.Header.Set("X-Api-Key", testKey)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var ids []string
	for _, m := range decode(t, resp.Body)["data"].([]any) {
		ids = append(ids, m.(map[string]any)["id"].(string))
	}
	if strings.Join(ids, ",") != "broken,flaky,picky,steady" {
		t.Fatalf("models = %v", ids)
	}
}
