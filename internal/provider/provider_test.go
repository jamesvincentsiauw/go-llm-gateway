package provider

import (
	"context"
	"errors"
	"fmt"
	"io"
	"testing"
)

func TestIsRetryable(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"canceled", context.Canceled, false},
		{"wrapped canceled", fmt.Errorf("read: %w", context.Canceled), false},
		{"deadline", context.DeadlineExceeded, true},
		{"dropped connection", io.ErrUnexpectedEOF, true},
		{"retryable provider error", &Error{Status: 503, Retryable: true}, true},
		{"client error", &Error{Status: 400, Retryable: false}, false},
		{"wrapped provider error", fmt.Errorf("open: %w", &Error{Status: 422}), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsRetryable(tt.err); got != tt.want {
				t.Fatalf("IsRetryable(%v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}

func TestRetryableStatus(t *testing.T) {
	for status, want := range map[int]bool{400: false, 401: true, 403: true, 404: false, 422: false, 408: true, 409: true, 429: true, 500: true, 529: true} {
		if got := RetryableStatus(status); got != want {
			t.Errorf("RetryableStatus(%d) = %v, want %v", status, got, want)
		}
	}
}

func TestErrorMessage(t *testing.T) {
	err := &Error{Provider: "openai", Status: 429, Err: errors.New("slow down")}
	if got, want := err.Error(), "openai: status 429: slow down"; got != want {
		t.Fatalf("Error() = %q, want %q", got, want)
	}
}
