// Package provider defines the upstream interface the gateway streams through.
package provider

import (
	"context"
	"errors"
	"fmt"
)

// Message is one chat turn. The gateway only handles text content.
type Message struct {
	Role    string // system, user or assistant
	Content string
}

// Request is a provider-neutral chat request.
type Request struct {
	Model       string
	Messages    []Message
	MaxTokens   int
	Temperature *float64
	TopP        *float64
}

// Finish reasons, using the OpenAI vocabulary the gateway exposes.
const (
	FinishStop          = "stop"
	FinishLength        = "length"
	FinishContentFilter = "content_filter"
)

// Usage counts tokens for one upstream call.
type Usage struct {
	InputTokens  int
	OutputTokens int
}

// Chunk is one piece of a streamed response. A chunk carries text, a finish
// reason, usage, or nothing at all as a sign that the upstream is still working.
type Chunk struct {
	Text         string
	FinishReason string
	Usage        *Usage
}

// Stream yields chunks until Next returns false. Err reports why it stopped
// early; it is nil when the upstream finished normally.
type Stream interface {
	Next() bool
	Chunk() Chunk
	Err() error
	Close() error
}

// Provider opens streaming chat completions against one upstream API.
type Provider interface {
	Name() string
	Stream(ctx context.Context, req Request) (Stream, error)
}

// Error is an upstream failure the gateway can reason about.
type Error struct {
	Provider  string
	Status    int // HTTP status, 0 for transport errors
	Retryable bool
	Err       error
}

func (e *Error) Error() string {
	if e.Status != 0 {
		return fmt.Sprintf("%s: status %d: %v", e.Provider, e.Status, e.Err)
	}
	return fmt.Sprintf("%s: %v", e.Provider, e.Err)
}

func (e *Error) Unwrap() error { return e.Err }

// RetryableStatus reports whether another provider may succeed where this
// status failed. That includes 401 and 403, which point at this provider's
// credentials rather than at the request.
func RetryableStatus(status int) bool {
	switch status {
	case 401, 403, 408, 409, 429:
		return true
	}
	return status >= 500
}

// IsRetryable reports whether err is worth sending to the next provider.
// Errors that are not *Error, such as a dropped connection mid-stream, count
// as retryable; context cancellation does not.
func IsRetryable(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) {
		return false
	}
	var perr *Error
	if errors.As(err, &perr) {
		return perr.Retryable
	}
	return true
}
