package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// OpenAI streams from any server that speaks the OpenAI chat completions API.
type OpenAI struct {
	name    string
	baseURL string
	apiKey  string
	client  *http.Client
}

// NewOpenAI returns a provider for baseURL, e.g. https://api.openai.com/v1.
func NewOpenAI(name, baseURL, apiKey string, client *http.Client) *OpenAI {
	if client == nil {
		client = http.DefaultClient
	}
	return &OpenAI{name: name, baseURL: strings.TrimSuffix(baseURL, "/"), apiKey: apiKey, client: client}
}

func (p *OpenAI) Name() string { return p.name }

type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIRequest struct {
	Model         string          `json:"model"`
	Messages      []openAIMessage `json:"messages"`
	MaxTokens     int             `json:"max_completion_tokens,omitempty"`
	Temperature   *float64        `json:"temperature,omitempty"`
	TopP          *float64        `json:"top_p,omitempty"`
	Stream        bool            `json:"stream"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
}

func (p *OpenAI) Stream(ctx context.Context, req Request) (Stream, error) {
	body := openAIRequest{
		Model:       req.Model,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      true,
	}
	body.StreamOptions.IncludeUsage = true
	for _, m := range req.Messages {
		body.Messages = append(body.Messages, openAIMessage(m))
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, &Error{Provider: p.name, Err: err}
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return nil, &Error{Provider: p.name, Err: err}
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, &Error{Provider: p.name, Retryable: ctx.Err() == nil, Err: err}
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, &Error{
			Provider:  p.name,
			Status:    resp.StatusCode,
			Retryable: RetryableStatus(resp.StatusCode),
			Err:       errors.New(strings.TrimSpace(string(msg))),
		}
	}
	return &openAIStream{name: p.name, body: resp.Body, scanner: newSSEScanner(resp.Body)}, nil
}

type openAIStream struct {
	name    string
	body    io.ReadCloser
	scanner *bufio.Scanner
	chunk   Chunk
	err     error
	done    bool
}

type openAIEvent struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func (s *openAIStream) Next() bool {
	if s.done {
		return false
	}
	for s.scanner.Scan() {
		data, ok := strings.CutPrefix(s.scanner.Text(), "data:")
		if !ok {
			continue
		}
		data = strings.TrimSpace(data)
		if data == "[DONE]" {
			s.done = true
			return false
		}
		var ev openAIEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			s.err = &Error{Provider: s.name, Retryable: true, Err: fmt.Errorf("decode event: %w", err)}
			return false
		}
		if ev.Error != nil {
			s.err = &Error{Provider: s.name, Retryable: true, Err: errors.New(ev.Error.Message)}
			return false
		}

		var c Chunk
		for _, choice := range ev.Choices {
			c.Text += choice.Delta.Content
			if choice.FinishReason != "" {
				c.FinishReason = choice.FinishReason
			}
		}
		if ev.Usage != nil {
			c.Usage = &Usage{InputTokens: ev.Usage.PromptTokens, OutputTokens: ev.Usage.CompletionTokens}
		}
		if c != (Chunk{}) {
			s.chunk = c
			return true
		}
	}
	// The body ended without [DONE], so the upstream went away mid-answer.
	s.err = s.scanner.Err()
	if s.err == nil {
		s.err = io.ErrUnexpectedEOF
	}
	s.err = &Error{Provider: s.name, Retryable: !errors.Is(s.err, context.Canceled), Err: s.err}
	return false
}

func (s *openAIStream) Chunk() Chunk { return s.chunk }
func (s *openAIStream) Err() error   { return s.err }
func (s *openAIStream) Close() error { return s.body.Close() }

func newSSEScanner(r io.Reader) *bufio.Scanner {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	return sc
}
