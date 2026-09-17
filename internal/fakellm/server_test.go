package fakellm

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type result struct {
	status int
	text   string
	usage  map[string]int
	err    error
}

func call(t *testing.T, srv *httptest.Server, body string) result {
	t.Helper()
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	res := result{status: resp.StatusCode}
	if resp.StatusCode != http.StatusOK {
		return res
	}
	sc := bufio.NewScanner(resp.Body)
	var sb strings.Builder
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var ev struct {
			Choices []struct {
				Delta struct{ Content string } `json:"delta"`
			} `json:"choices"`
			Usage map[string]int `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			t.Fatal(err)
		}
		for _, c := range ev.Choices {
			sb.WriteString(c.Delta.Content)
		}
		if ev.Usage != nil {
			res.usage = ev.Usage
		}
	}
	res.text = sb.String()
	res.err = sc.Err()
	return res
}

func TestStreamsPassage(t *testing.T) {
	srv := httptest.NewServer(Handler())
	defer srv.Close()

	res := call(t, srv, `{"model":"fake","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi there"}]}`)
	if res.err != nil || res.text != Passage {
		t.Fatalf("got %q (err %v), want the full passage", res.text, res.err)
	}
	if res.usage["prompt_tokens"] != 2 || res.usage["completion_tokens"] != len(strings.Fields(Passage)) {
		t.Fatalf("usage = %v", res.usage)
	}
}

func TestCutDropsConnection(t *testing.T) {
	srv := httptest.NewServer(Handler())
	defer srv.Close()

	res := call(t, srv, `{"model":"fake-cut-3","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	want := strings.Join(strings.Fields(Passage)[:3], " ")
	if res.text != want {
		t.Fatalf("got %q, want %q", res.text, want)
	}
}

func TestContinuationResumesAfterAssistantTurn(t *testing.T) {
	srv := httptest.NewServer(Handler())
	defer srv.Close()

	words := strings.Fields(Passage)
	partial := strings.Join(words[:5], " ")
	body := `{"model":"fake","stream":true,"messages":[` +
		`{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":"` + partial + `"},` +
		`{"role":"user","content":"continue"}]}`

	res := call(t, srv, body)
	if got := partial + res.text; got != Passage {
		t.Fatalf("partial + continuation = %q, want the passage", got)
	}
}

func TestScriptedStatus(t *testing.T) {
	srv := httptest.NewServer(Handler())
	defer srv.Close()

	if res := call(t, srv, `{"model":"fake-status-503","stream":true,"messages":[]}`); res.status != 503 {
		t.Fatalf("status = %d, want 503", res.status)
	}
	if res := call(t, srv, `{"model":"gpt-x","stream":true,"messages":[]}`); res.status != 404 {
		t.Fatalf("unknown model status = %d, want 404", res.status)
	}
	if res := call(t, srv, `{"model":"fake","stream":false,"messages":[]}`); res.status != 400 {
		t.Fatalf("non-stream status = %d, want 400", res.status)
	}
}

func TestCutSurfacesAsReadError(t *testing.T) {
	srv := httptest.NewServer(Handler())
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"fake-cut-2","stream":true,"messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if _, err := io.ReadAll(resp.Body); err == nil {
		t.Fatal("expected a read error from the dropped connection")
	}
}
