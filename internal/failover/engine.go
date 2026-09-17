// Package failover streams a request through an ordered list of upstreams and
// moves to the next one when an upstream fails, including in the middle of an
// answer.
package failover

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/jamesvincentsiauw/go-llm-gateway/internal/breaker"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/provider"
)

// ContinuationPrompt asks the next upstream to finish an interrupted answer.
// Current Claude models reject assistant prefill, so the partial answer is sent
// as a finished assistant turn followed by this instruction.
const ContinuationPrompt = "Your previous reply was cut off. Continue it from exactly where it stops. " +
	"Do not repeat any of it, and do not add an introduction or acknowledge the interruption."

// Attempt outcomes reported to the Observer.
const (
	OutcomeOK          = "ok"
	OutcomeFailed      = "failed"
	OutcomeStalled     = "stalled"
	OutcomeRejected    = "rejected"
	OutcomeBreakerOpen = "breaker_open"
)

var (
	// ErrExhausted means every upstream was skipped or failed.
	ErrExhausted = errors.New("all upstreams failed")
	errStalled   = errors.New("upstream stalled")
)

// Target is one upstream model to try.
type Target struct {
	Provider provider.Provider
	Model    string
	Breaker  *breaker.Breaker
}

// Sink receives what the client should see.
type Sink interface {
	Text(s string) error
	Failover(from, to string, midStream bool) error
}

// Observer receives per-attempt telemetry.
type Observer interface {
	Attempt(provider, outcome string)
	FailedOver(from, to string, midStream bool)
}

// Result describes a completed request.
type Result struct {
	FinishReason string
	Usage        provider.Usage // summed over every attempt that reported usage
	Providers    []string       // upstreams that contributed text, in order
}

// Engine runs requests. StallTimeout bounds both the wait for the first chunk
// and the gap between chunks; zero disables it.
type Engine struct {
	StallTimeout time.Duration
	Observer     Observer
	Log          *slog.Logger
}

// Run streams req through targets in order and returns once one of them
// finishes the answer.
func (e *Engine) Run(ctx context.Context, req provider.Request, targets []Target, sink Sink) (Result, error) {
	run := &runState{engine: e, req: req, sink: sink}
	var lastErr error
	prev := ""

	for _, t := range targets {
		name := t.Provider.Name()
		if !t.Breaker.Allow() {
			e.Observer.Attempt(name, OutcomeBreakerOpen)
			continue
		}
		if prev != "" {
			midStream := run.written.Len() > 0
			e.Observer.FailedOver(prev, name, midStream)
			e.Log.Warn("failing over", "from", prev, "to", name, "mid_stream", midStream, "err", lastErr)
			if err := sink.Failover(prev, name, midStream); err != nil {
				return run.result, err
			}
		}

		done, err := run.attempt(ctx, t)
		switch {
		case done:
			t.Breaker.Success()
			e.Observer.Attempt(name, OutcomeOK)
			return run.result, nil
		case ctx.Err() != nil || errors.Is(err, errSinkFailed):
			t.Breaker.Release()
			return run.result, err
		case errors.Is(err, errStalled):
			t.Breaker.Failure()
			e.Observer.Attempt(name, OutcomeStalled)
		case provider.IsRetryable(err):
			t.Breaker.Failure()
			e.Observer.Attempt(name, OutcomeFailed)
		default:
			// The request itself was refused; another upstream would refuse it too.
			t.Breaker.Release()
			e.Observer.Attempt(name, OutcomeRejected)
			return run.result, err
		}
		lastErr, prev = err, name
	}

	if lastErr == nil {
		return run.result, fmt.Errorf("%w: every circuit breaker is open", ErrExhausted)
	}
	return run.result, fmt.Errorf("%w: %w", ErrExhausted, lastErr)
}

var errSinkFailed = errors.New("client write failed")

type runState struct {
	engine  *Engine
	req     provider.Request
	sink    Sink
	written strings.Builder // text the client has received
	result  Result
}

// attempt streams one upstream. It returns done=true when the upstream
// finished the answer.
func (r *runState) attempt(ctx context.Context, t Target) (done bool, err error) {
	req, ok := r.requestFor(t.Model)
	if !ok {
		r.result.FinishReason = provider.FinishLength
		return true, nil
	}

	actx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	watchdog := r.engine.watchdog(cancel)
	defer watchdog.stop()

	stream, err := t.Provider.Stream(actx, req)
	if err != nil {
		return false, stallOr(actx, err)
	}
	defer stream.Close()

	r.result.FinishReason = ""
	var hold string // text received but not yet sent, from the last whitespace on
	continuing := r.written.Len() > 0
	contributed := false
	for stream.Next() {
		watchdog.reset()
		c := stream.Chunk()
		if c.Usage != nil {
			r.result.Usage.InputTokens += c.Usage.InputTokens
			r.result.Usage.OutputTokens += c.Usage.OutputTokens
		}
		if c.FinishReason != "" {
			r.result.FinishReason = c.FinishReason
		}
		if c.Text == "" {
			continue
		}
		text := c.Text
		if continuing && hold == "" && !contributed && needsSpace(r.written.String(), text) {
			text = " " + text
		}
		hold += text

		// Send everything up to the last whitespace so a failover never splits a word.
		if cut := lastSpace(hold); cut > 0 {
			if err := r.send(hold[:cut]); err != nil {
				return false, err
			}
			hold = hold[cut:]
			if !contributed {
				contributed = true
				r.result.Providers = append(r.result.Providers, t.Provider.Name())
			}
		}
	}
	if err := stream.Err(); err != nil {
		// hold is dropped: the next upstream rewrites from the last full word.
		return false, stallOr(actx, err)
	}

	if hold != "" {
		if err := r.send(hold); err != nil {
			return false, err
		}
		if !contributed {
			r.result.Providers = append(r.result.Providers, t.Provider.Name())
		}
	}
	if r.result.FinishReason == "" {
		r.result.FinishReason = provider.FinishStop
	}
	return true, nil
}

// requestFor builds the request for model, turning it into a continuation when
// the client already has part of the answer. ok is false when the token budget
// is already spent.
func (r *runState) requestFor(model string) (provider.Request, bool) {
	req := r.req
	req.Model = model
	if r.written.Len() == 0 {
		return req, true
	}

	remaining := r.req.MaxTokens - EstimateTokens(r.written.String())
	if remaining <= 0 {
		return req, false
	}
	req.MaxTokens = remaining
	req.Messages = append(append([]provider.Message(nil), r.req.Messages...),
		provider.Message{Role: "assistant", Content: r.written.String()},
		provider.Message{Role: "user", Content: ContinuationPrompt},
	)
	return req, true
}

func (r *runState) send(s string) error {
	if err := r.sink.Text(s); err != nil {
		return fmt.Errorf("%w: %w", errSinkFailed, err)
	}
	r.written.WriteString(s)
	return nil
}

// EstimateTokens approximates a token count at four bytes per token.
func EstimateTokens(s string) int {
	return (len(s) + 3) / 4
}

func lastSpace(s string) int {
	return strings.LastIndexFunc(s, unicode.IsSpace)
}

// needsSpace reports whether a continuation that starts with next should be
// separated from prev, for models that drop the leading space.
func needsSpace(prev, next string) bool {
	last, _ := utf8.DecodeLastRuneInString(prev)
	first, _ := utf8.DecodeRuneInString(next)
	return !unicode.IsSpace(last) && !unicode.IsSpace(first) && !unicode.IsPunct(first)
}

func stallOr(ctx context.Context, err error) error {
	if errors.Is(context.Cause(ctx), errStalled) {
		return fmt.Errorf("%w: %w", errStalled, err)
	}
	return err
}

type watchdog struct {
	mu      sync.Mutex
	timer   *time.Timer
	timeout time.Duration
}

func (e *Engine) watchdog(cancel context.CancelCauseFunc) *watchdog {
	w := &watchdog{timeout: e.StallTimeout}
	if w.timeout > 0 {
		w.timer = time.AfterFunc(w.timeout, func() { cancel(errStalled) })
	}
	return w
}

func (w *watchdog) reset() {
	if w.timer == nil {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.timer.Reset(w.timeout)
}

func (w *watchdog) stop() {
	if w.timer != nil {
		w.timer.Stop()
	}
}
