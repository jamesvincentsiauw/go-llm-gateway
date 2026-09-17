package provider_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jamesvincentsiauw/go-llm-gateway/internal/fakellm"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/provider"
)

func drain(t *testing.T, s provider.Stream) (text, finish string, usage *provider.Usage, err error) {
	t.Helper()
	defer s.Close()
	var sb strings.Builder
	for s.Next() {
		c := s.Chunk()
		sb.WriteString(c.Text)
		if c.FinishReason != "" {
			finish = c.FinishReason
		}
		if c.Usage != nil {
			usage = c.Usage
		}
	}
	return sb.String(), finish, usage, s.Err()
}

func fakeOpenAI(t *testing.T) *provider.OpenAI {
	t.Helper()
	srv := httptest.NewServer(fakellm.Handler())
	t.Cleanup(srv.Close)
	return provider.NewOpenAI("fake", srv.URL+"/v1", "secret", srv.Client())
}

func userRequest(model string) provider.Request {
	return provider.Request{Model: model, MaxTokens: 100, Messages: []provider.Message{{Role: "user", Content: "tell me"}}}
}

func TestOpenAIStreamsFullResponse(t *testing.T) {
	s, err := fakeOpenAI(t).Stream(context.Background(), userRequest("fake"))
	if err != nil {
		t.Fatal(err)
	}
	text, finish, usage, err := drain(t, s)
	if err != nil {
		t.Fatal(err)
	}
	if text != fakellm.Passage || finish != provider.FinishStop {
		t.Fatalf("text = %q, finish = %q", text, finish)
	}
	if usage == nil || usage.InputTokens != 2 || usage.OutputTokens != len(strings.Fields(fakellm.Passage)) {
		t.Fatalf("usage = %+v", usage)
	}
}

func TestOpenAIDroppedStreamIsRetryable(t *testing.T) {
	s, err := fakeOpenAI(t).Stream(context.Background(), userRequest("fake-cut-4"))
	if err != nil {
		t.Fatal(err)
	}
	text, _, _, err := drain(t, s)
	if err == nil {
		t.Fatal("expected an error from the dropped stream")
	}
	if !provider.IsRetryable(err) {
		t.Fatalf("dropped stream should be retryable: %v", err)
	}
	if want := strings.Join(strings.Fields(fakellm.Passage)[:4], " "); text != want {
		t.Fatalf("text before the drop = %q, want %q", text, want)
	}
}

func TestOpenAIStatusErrors(t *testing.T) {
	p := fakeOpenAI(t)
	tests := map[string]bool{"fake-status-503": true, "fake-status-429": true, "fake-status-400": false, "fake-status-401": false}
	for model, retryable := range tests {
		_, err := p.Stream(context.Background(), userRequest(model))
		var perr *provider.Error
		if !errors.As(err, &perr) {
			t.Fatalf("%s: err = %v, want *provider.Error", model, err)
		}
		if perr.Retryable != retryable {
			t.Errorf("%s: retryable = %v, want %v", model, perr.Retryable, retryable)
		}
	}
}

func TestOpenAISendsCredentialsAndBody(t *testing.T) {
	var gotAuth, gotBody string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	temp := 0.2
	req := userRequest("gpt-test")
	req.Temperature = &temp
	s, err := provider.NewOpenAI("openai", srv.URL, "sk-test", nil).Stream(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := drain(t, s); err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer sk-test" {
		t.Errorf("Authorization = %q", gotAuth)
	}
	for _, want := range []string{`"model":"gpt-test"`, `"stream":true`, `"include_usage":true`, `"max_completion_tokens":100`, `"temperature":0.2`} {
		if !strings.Contains(gotBody, want) {
			t.Errorf("body %s missing %s", gotBody, want)
		}
	}
}
