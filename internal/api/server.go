// Package api serves an OpenAI-compatible chat completions endpoint.
package api

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/jamesvincentsiauw/go-llm-gateway/internal/failover"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/metrics"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/provider"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/quota"
)

const maxBodyBytes = 10 << 20

// Client is a caller of the gateway.
type Client struct {
	Name            string
	DailyTokenLimit int
}

// Server handles gateway requests. Clients are keyed by the SHA-256 of their
// API key so raw keys are not kept in memory after startup.
type Server struct {
	Routes           map[string][]failover.Target
	Clients          map[[32]byte]Client
	Engine           *failover.Engine
	Quota            *quota.Limiter
	DefaultMaxTokens int
	MaxTokensLimit   int
	Metrics          *metrics.Metrics
	Log              *slog.Logger
}

// HashKey returns the lookup key for an API key.
func HashKey(key string) [32]byte { return sha256.Sum256([]byte(key)) }

// Handler returns the HTTP routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.chat)
	mux.HandleFunc("GET /v1/models", s.models)
	return mux
}

type chatRequest struct {
	Model               string        `json:"model"`
	Messages            []chatMessage `json:"messages"`
	MaxTokens           *int          `json:"max_tokens"`
	MaxCompletionTokens *int          `json:"max_completion_tokens"`
	Temperature         *float64      `json:"temperature"`
	TopP                *float64      `json:"top_p"`
	Stream              bool          `json:"stream"`
	StreamOptions       *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
}

type chatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// apiError is written as {"error": {...}} with the given HTTP status.
type apiError struct {
	Status  int    `json:"-"`
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

func (e *apiError) Error() string { return e.Message }

func (s *Server) chat(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	route, status := "unknown", http.StatusOK
	defer func() {
		s.Metrics.Requests.WithLabelValues(route, fmt.Sprint(status)).Inc()
		s.Metrics.Duration.WithLabelValues(route).Observe(time.Since(start).Seconds())
	}()
	fail := func(e *apiError) {
		status = e.Status
		writeError(w, e)
	}

	client, ok := s.authenticate(r)
	if !ok {
		fail(&apiError{Status: http.StatusUnauthorized, Type: "invalid_request_error", Code: "invalid_api_key", Message: "missing or unknown API key"})
		return
	}

	var body chatRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBodyBytes)).Decode(&body); err != nil {
		fail(invalid("request body is not valid JSON: " + err.Error()))
		return
	}
	targets, ok := s.Routes[body.Model]
	if !ok {
		fail(&apiError{Status: http.StatusNotFound, Type: "invalid_request_error", Code: "model_not_found",
			Message: fmt.Sprintf("model %q is not configured on this gateway", body.Model)})
		return
	}
	route = body.Model

	req, apiErr := s.toProviderRequest(body)
	if apiErr != nil {
		fail(apiErr)
		return
	}

	inputEstimate := 0
	for _, m := range req.Messages {
		inputEstimate += failover.EstimateTokens(m.Content)
	}
	res, err := s.Quota.Reserve(r.Context(), client.Name, inputEstimate+req.MaxTokens, client.DailyTokenLimit)
	switch {
	case errors.Is(err, quota.ErrExceeded):
		fail(&apiError{Status: http.StatusTooManyRequests, Type: "insufficient_quota", Code: "daily_token_limit", Message: err.Error()})
		return
	case err != nil:
		s.Log.Error("quota unavailable", "err", err)
		fail(&apiError{Status: http.StatusServiceUnavailable, Type: "server_error", Message: "quota store unavailable"})
		return
	}
	if res.Remaining >= 0 {
		w.Header().Set("X-Gateway-Quota-Remaining", fmt.Sprint(res.Remaining))
	}

	var sink textSink
	meta := responseMeta{id: "chatcmpl-" + rand.Text(), model: body.Model, created: time.Now().Unix()}
	if body.Stream {
		sink = &streamSink{w: w, meta: meta, includeUsage: body.StreamOptions != nil && body.StreamOptions.IncludeUsage}
	} else {
		sink = &bufferSink{}
	}

	result, runErr := s.Engine.Run(r.Context(), req, targets, sink)

	used := result.Usage.InputTokens + result.Usage.OutputTokens
	if used == 0 && sink.written() > 0 {
		// Upstreams that dropped mid-answer never reported usage.
		used = inputEstimate + failover.TokensForBytes(sink.written())
	}
	settleCtx, cancel := context.WithTimeout(context.WithoutCancel(r.Context()), 2*time.Second)
	defer cancel()
	if err := res.Settle(settleCtx, used); err != nil {
		s.Log.Error("settle quota", "client", client.Name, "err", err)
	}
	s.Metrics.Tokens.WithLabelValues(client.Name, "input").Add(float64(result.Usage.InputTokens))
	s.Metrics.Tokens.WithLabelValues(client.Name, "output").Add(float64(result.Usage.OutputTokens))

	log := s.Log.With("client", client.Name, "route", route, "upstreams", result.Providers, "finish", result.FinishReason)
	if runErr != nil {
		apiErr := toAPIError(runErr)
		status = apiErr.Status
		if r.Context().Err() != nil {
			status = 499 // client closed the request
		}
		log.Warn("request failed", "err", runErr)
		sink.fail(w, apiErr)
		return
	}
	log.Info("request served", "input_tokens", result.Usage.InputTokens, "output_tokens", result.Usage.OutputTokens)
	sink.finish(w, meta, result)
}

func (s *Server) toProviderRequest(body chatRequest) (provider.Request, *apiError) {
	req := provider.Request{Temperature: body.Temperature, TopP: body.TopP, MaxTokens: s.DefaultMaxTokens}
	switch {
	case body.MaxCompletionTokens != nil:
		req.MaxTokens = *body.MaxCompletionTokens
	case body.MaxTokens != nil:
		req.MaxTokens = *body.MaxTokens
	}
	if req.MaxTokens < 1 || req.MaxTokens > s.MaxTokensLimit {
		return req, invalid(fmt.Sprintf("max_tokens must be between 1 and %d", s.MaxTokensLimit))
	}
	if len(body.Messages) == 0 {
		return req, invalid("messages must not be empty")
	}
	for i, m := range body.Messages {
		role := m.Role
		if role == "developer" {
			role = "system"
		}
		if role != "system" && role != "user" && role != "assistant" {
			return req, invalid(fmt.Sprintf("messages[%d]: role %q is not supported", i, m.Role))
		}
		text, err := textContent(m.Content)
		if err != nil {
			return req, invalid(fmt.Sprintf("messages[%d]: %v", i, err))
		}
		req.Messages = append(req.Messages, provider.Message{Role: role, Content: text})
	}
	return req, nil
}

// textContent accepts a string or an array of text parts.
func textContent(raw json.RawMessage) (string, error) {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", errors.New("content must be a string or an array of parts")
	}
	var sb strings.Builder
	for _, p := range parts {
		if p.Type != "text" {
			return "", fmt.Errorf("content part type %q is not supported, only text", p.Type)
		}
		sb.WriteString(p.Text)
	}
	return sb.String(), nil
}

func (s *Server) authenticate(r *http.Request) (Client, bool) {
	key, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
	if !ok {
		key = r.Header.Get("X-Api-Key")
	}
	if key == "" {
		return Client{}, false
	}
	c, ok := s.Clients[HashKey(strings.TrimSpace(key))]
	return c, ok
}

func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.authenticate(r); !ok {
		writeError(w, &apiError{Status: http.StatusUnauthorized, Type: "invalid_request_error", Code: "invalid_api_key", Message: "missing or unknown API key"})
		return
	}
	type model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		OwnedBy string `json:"owned_by"`
	}
	data := []model{}
	for id := range s.Routes {
		data = append(data, model{ID: id, Object: "model", OwnedBy: "gateway"})
	}
	slices.SortFunc(data, func(a, b model) int { return strings.Compare(a.ID, b.ID) })
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

func invalid(msg string) *apiError {
	return &apiError{Status: http.StatusBadRequest, Type: "invalid_request_error", Message: msg}
}

// toAPIError maps an engine error to what the client sees.
func toAPIError(err error) *apiError {
	var perr *provider.Error
	switch {
	case errors.Is(err, failover.ErrExhausted):
		return &apiError{Status: http.StatusServiceUnavailable, Type: "upstream_unavailable", Message: "every upstream for this model failed"}
	case errors.As(err, &perr) && perr.Status >= 400 && perr.Status < 500 && perr.Status != 401 && perr.Status != 403:
		return &apiError{Status: http.StatusBadRequest, Type: "invalid_request_error", Message: "upstream rejected the request: " + perr.Err.Error()}
	default:
		// Upstream auth failures are a gateway misconfiguration, not the caller's fault.
		return &apiError{Status: http.StatusBadGateway, Type: "upstream_error", Message: "upstream request failed"}
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, e *apiError) {
	writeJSON(w, e.Status, map[string]any{"error": e})
}
