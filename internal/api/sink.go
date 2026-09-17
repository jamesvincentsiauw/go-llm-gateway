package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/jamesvincentsiauw/go-llm-gateway/internal/failover"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/provider"
)

type responseMeta struct {
	id      string
	model   string
	created int64
}

// textSink is a failover.Sink that also knows how to end the HTTP response.
type textSink interface {
	failover.Sink
	written() int
	finish(w http.ResponseWriter, meta responseMeta, res failover.Result)
	fail(w http.ResponseWriter, e *apiError)
}

func usageJSON(u provider.Usage) map[string]int {
	return map[string]int{
		"prompt_tokens":     u.InputTokens,
		"completion_tokens": u.OutputTokens,
		"total_tokens":      u.InputTokens + u.OutputTokens,
	}
}

// bufferSink collects the answer for a non-streaming response.
type bufferSink struct {
	text      strings.Builder
	failovers []string
}

func (b *bufferSink) Text(s string) error { b.text.WriteString(s); return nil }

func (b *bufferSink) Failover(from, to string, midStream bool) error {
	b.failovers = append(b.failovers, from+"->"+to)
	return nil
}

func (b *bufferSink) written() int { return b.text.Len() }

func (b *bufferSink) finish(w http.ResponseWriter, meta responseMeta, res failover.Result) {
	w.Header().Set("X-Gateway-Upstreams", strings.Join(res.Providers, ","))
	writeJSON(w, http.StatusOK, map[string]any{
		"id":      meta.id,
		"object":  "chat.completion",
		"created": meta.created,
		"model":   meta.model,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": b.text.String()},
			"finish_reason": res.FinishReason,
		}},
		"usage":     usageJSON(res.Usage),
		"x_gateway": map[string]any{"upstreams": res.Providers, "failovers": b.failovers},
	})
}

func (b *bufferSink) fail(w http.ResponseWriter, e *apiError) { writeError(w, e) }

// streamSink writes server-sent events in the OpenAI chunk format. Headers go
// out with the first text, so a request that fails before any output still
// gets a normal JSON error and status code.
type streamSink struct {
	w            http.ResponseWriter
	meta         responseMeta
	includeUsage bool
	started      bool
	bytes        int
}

func (s *streamSink) start() error {
	if s.started {
		return nil
	}
	s.started = true
	h := s.w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	s.w.WriteHeader(http.StatusOK)
	return s.chunk(map[string]any{"role": "assistant", "content": ""}, nil, nil)
}

func (s *streamSink) chunk(delta map[string]any, finish any, usage any) error {
	ev := map[string]any{
		"id":      s.meta.id,
		"object":  "chat.completion.chunk",
		"created": s.meta.created,
		"model":   s.meta.model,
		"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
	}
	if usage != nil {
		ev["choices"] = []any{}
		ev["usage"] = usage
	}
	b, err := json.Marshal(ev)
	if err != nil {
		return err
	}
	return s.write("data: " + string(b) + "\n\n")
}

func (s *streamSink) write(line string) error {
	if _, err := fmt.Fprint(s.w, line); err != nil {
		return err
	}
	return http.NewResponseController(s.w).Flush()
}

func (s *streamSink) Text(text string) error {
	if err := s.start(); err != nil {
		return err
	}
	s.bytes += len(text)
	return s.chunk(map[string]any{"content": text}, nil, nil)
}

// Failover is reported as an SSE comment, which OpenAI clients ignore.
func (s *streamSink) Failover(from, to string, midStream bool) error {
	if !s.started {
		return nil
	}
	return s.write(fmt.Sprintf(": gateway failover from=%s to=%s\n\n", from, to))
}

func (s *streamSink) written() int { return s.bytes }

func (s *streamSink) finish(w http.ResponseWriter, meta responseMeta, res failover.Result) {
	if err := s.start(); err != nil {
		return
	}
	_ = s.chunk(map[string]any{}, res.FinishReason, nil)
	if s.includeUsage {
		_ = s.chunk(nil, nil, usageJSON(res.Usage))
	}
	_ = s.write("data: [DONE]\n\n")
}

func (s *streamSink) fail(w http.ResponseWriter, e *apiError) {
	if !s.started {
		writeError(w, e)
		return
	}
	b, _ := json.Marshal(map[string]any{"error": e})
	_ = s.write("data: " + string(b) + "\n\n")
}
