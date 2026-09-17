// Package fakellm is an OpenAI-compatible chat server with scripted failures,
// used by tests and the local demo so neither needs a real API key.
//
// The model name picks the behavior:
//
//	fake            streams Passage word by word
//	fake-cut-N      drops the connection after N words
//	fake-status-N   answers with HTTP status N before streaming
//	fake-slow-D     waits D between words, e.g. fake-slow-50ms
//
// If the request already contains an assistant turn, the server treats it as a
// continuation and resumes Passage right after the words that turn holds.
package fakellm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Passage is the text every fake model produces.
const Passage = "Retries hide failures until they do not. A gateway that can switch providers " +
	"in the middle of an answer keeps the reader from noticing that the first upstream " +
	"went away, as long as the second one picks up at the right word."

type chatRequest struct {
	Model    string `json:"model"`
	Stream   bool   `json:"stream"`
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
	StreamOptions *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
}

// Handler serves POST /v1/chat/completions.
func Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", serve)
	return mux
}

func serve(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	if !req.Stream {
		writeError(w, http.StatusBadRequest, "fakellm only supports stream=true")
		return
	}

	if req.Model != "fake" && !strings.HasPrefix(req.Model, "fake-") {
		writeError(w, http.StatusNotFound, "unknown model "+req.Model)
		return
	}
	mode, arg, _ := strings.Cut(strings.TrimPrefix(req.Model, "fake-"), "-")
	if req.Model == "fake" {
		mode = ""
	}
	cutAfter, delay := -1, time.Duration(0)
	switch mode {
	case "":
	case "cut":
		n, err := strconv.Atoi(arg)
		if err != nil {
			writeError(w, http.StatusBadRequest, "fake-cut-N needs a number")
			return
		}
		cutAfter = n
	case "status":
		code, err := strconv.Atoi(arg)
		if err != nil {
			writeError(w, http.StatusBadRequest, "fake-status-N needs a number")
			return
		}
		writeError(w, code, "scripted failure")
		return
	case "slow":
		d, err := time.ParseDuration(arg)
		if err != nil {
			writeError(w, http.StatusBadRequest, "fake-slow-D needs a duration")
			return
		}
		delay = d
	default:
		writeError(w, http.StatusNotFound, "unknown model "+req.Model)
		return
	}

	words := strings.Fields(Passage)
	inputWords, alreadyWritten := 0, 0
	for _, m := range req.Messages {
		inputWords += len(strings.Fields(m.Content))
		if m.Role == "assistant" {
			alreadyWritten += len(strings.Fields(m.Content))
		}
	}
	words = words[min(alreadyWritten, len(words)):]

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)

	for i, word := range words {
		if i == cutAfter {
			abort(w)
			return
		}
		if delay > 0 {
			select {
			case <-time.After(delay):
			case <-r.Context().Done():
				return
			}
		}
		// Words after the first carry their leading space, like real token deltas.
		text := word
		if alreadyWritten+i > 0 {
			text = " " + word
		}
		writeEvent(w, map[string]any{
			"model":   req.Model,
			"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"content": text}}},
		})
		flusher.Flush()
	}

	writeEvent(w, map[string]any{
		"model":   req.Model,
		"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": "stop"}},
	})
	if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
		writeEvent(w, map[string]any{
			"model":   req.Model,
			"choices": []any{},
			"usage":   map[string]any{"prompt_tokens": inputWords, "completion_tokens": len(words)},
		})
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func writeEvent(w http.ResponseWriter, v any) {
	b, _ := json.Marshal(v)
	fmt.Fprintf(w, "data: %s\n\n", b)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": msg}})
}

// abort closes the connection without finishing the chunked response, which
// the client sees as an unexpected EOF.
func abort(w http.ResponseWriter) {
	if hj, ok := w.(http.Hijacker); ok {
		if conn, _, err := hj.Hijack(); err == nil {
			_ = conn.Close()
			return
		}
	}
	panic(http.ErrAbortHandler)
}
