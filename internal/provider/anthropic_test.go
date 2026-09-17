package provider_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jamesvincentsiauw/go-llm-gateway/internal/provider"
)

// claudeServer replays a Messages API event stream. It stops after cutAfter
// text deltas when cutAfter >= 0, and records the last request body.
type claudeServer struct {
	status   int
	words    []string
	cutAfter int
	stop     string
	lastBody map[string]any
}

func (c *claudeServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	_ = json.NewDecoder(r.Body).Decode(&c.lastBody)
	if c.status != 0 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(c.status)
		fmt.Fprint(w, `{"type":"error","error":{"type":"overloaded_error","message":"busy"}}`)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	send := func(event, data string) {
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data)
		w.(http.Flusher).Flush()
	}
	send("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"usage":{"input_tokens":12,"output_tokens":1}}}`)
	send("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)
	for i, word := range c.words {
		if i == c.cutAfter {
			return
		}
		b, _ := json.Marshal(word)
		send("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":`+string(b)+`}}`)
	}
	send("content_block_stop", `{"type":"content_block_stop","index":0}`)
	send("message_delta", `{"type":"message_delta","delta":{"stop_reason":"`+c.stop+`","stop_sequence":null},"usage":{"output_tokens":`+fmt.Sprint(len(c.words))+`}}`)
	send("message_stop", `{"type":"message_stop"}`)
}

func claude(t *testing.T, srv *claudeServer, fallbacks bool) provider.Provider {
	t.Helper()
	if srv.stop == "" {
		srv.stop = "end_turn"
	}
	hs := httptest.NewServer(srv)
	t.Cleanup(hs.Close)
	return provider.NewAnthropic("claude", provider.AnthropicOptions{APIKey: "test", BaseURL: hs.URL, RefusalFallbacks: fallbacks})
}

func TestAnthropicStreamsTextAndUsage(t *testing.T) {
	srv := &claudeServer{words: []string{"Hello", " from", " Claude"}, cutAfter: -1}
	temp := 0.7
	req := provider.Request{
		Model:       "claude-opus-5",
		MaxTokens:   64,
		Temperature: &temp,
		Messages: []provider.Message{
			{Role: "system", Content: "Be brief."},
			{Role: "user", Content: "hi"},
			{Role: "assistant", Content: "Hello"},
			{Role: "user", Content: "continue"},
		},
	}
	s, err := claude(t, srv, true).Stream(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	text, finish, usage, err := drain(t, s)
	if err != nil {
		t.Fatal(err)
	}
	if text != "Hello from Claude" || finish != provider.FinishStop {
		t.Fatalf("text = %q, finish = %q", text, finish)
	}
	if usage == nil || usage.InputTokens != 12 || usage.OutputTokens != 3 {
		t.Fatalf("usage = %+v", usage)
	}

	body := srv.lastBody
	if _, ok := body["temperature"]; ok {
		t.Error("temperature must not be sent to Claude")
	}
	if body["fallbacks"] != "default" {
		t.Errorf("fallbacks = %v, want default", body["fallbacks"])
	}
	if sys, _ := json.Marshal(body["system"]); !strings.Contains(string(sys), "Be brief.") {
		t.Errorf("system = %s", sys)
	}
	if msgs, _ := body["messages"].([]any); len(msgs) != 3 {
		t.Errorf("messages = %v, want 3 turns without the system prompt", body["messages"])
	}
}

func TestAnthropicMapsStopReasons(t *testing.T) {
	for stop, want := range map[string]string{"max_tokens": provider.FinishLength, "refusal": provider.FinishContentFilter} {
		s, err := claude(t, &claudeServer{words: []string{"x"}, cutAfter: -1, stop: stop}, false).
			Stream(context.Background(), userRequest("claude-opus-5"))
		if err != nil {
			t.Fatal(err)
		}
		if _, finish, _, _ := drain(t, s); finish != want {
			t.Errorf("stop %s mapped to %q, want %q", stop, finish, want)
		}
	}
}

func TestAnthropicDroppedStreamIsRetryable(t *testing.T) {
	srv := &claudeServer{words: []string{"one", " two", " three"}, cutAfter: 2}
	s, err := claude(t, srv, false).Stream(context.Background(), userRequest("claude-opus-5"))
	if err != nil {
		t.Fatal(err)
	}
	text, _, _, err := drain(t, s)
	if text != "one two" {
		t.Fatalf("text = %q", text)
	}
	if !provider.IsRetryable(err) || !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("err = %v, want retryable unexpected EOF", err)
	}
	if _, ok := srv.lastBody["fallbacks"]; ok {
		t.Error("fallbacks sent although disabled")
	}
}

func TestAnthropicStatusErrors(t *testing.T) {
	for status, retryable := range map[int]bool{529: true, 429: true, 400: false} {
		s, err := claude(t, &claudeServer{status: status}, false).Stream(context.Background(), userRequest("claude-opus-5"))
		if err != nil {
			t.Fatal(err)
		}
		_, _, _, err = drain(t, s)
		var perr *provider.Error
		if !errors.As(err, &perr) || perr.Status != status || perr.Retryable != retryable {
			t.Errorf("status %d: err = %v", status, err)
		}
	}
}
