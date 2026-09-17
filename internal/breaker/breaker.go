// Package breaker stops the gateway from sending traffic to a provider that
// keeps failing.
package breaker

import (
	"sync"
	"time"
)

// State of a breaker.
type State int

const (
	Closed   State = iota // traffic flows
	Open                  // traffic is skipped until the cooldown ends
	HalfOpen              // one trial request is allowed through
)

func (s State) String() string {
	switch s {
	case Open:
		return "open"
	case HalfOpen:
		return "half_open"
	default:
		return "closed"
	}
}

// Breaker opens after Threshold consecutive failures and allows a single trial
// request once Cooldown has passed.
type Breaker struct {
	threshold int
	cooldown  time.Duration
	now       func() time.Time

	mu       sync.Mutex
	state    State
	failures int
	openedAt time.Time
	trial    bool
}

// New returns a closed breaker.
func New(threshold int, cooldown time.Duration) *Breaker {
	return &Breaker{threshold: threshold, cooldown: cooldown, now: time.Now}
}

// Allow reports whether a request may be sent now. When it returns true for a
// half-open breaker, the caller owns the trial and must report its outcome.
func (b *Breaker) Allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()

	switch b.state {
	case Closed:
		return true
	case Open:
		if b.now().Sub(b.openedAt) < b.cooldown {
			return false
		}
		b.state = HalfOpen
		b.trial = true
		return true
	default:
		if b.trial {
			return false
		}
		b.trial = true
		return true
	}
}

// Success closes the breaker.
func (b *Breaker) Success() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.state, b.failures, b.trial = Closed, 0, false
}

// Failure counts a failed request and opens the breaker when needed.
func (b *Breaker) Failure() {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.failures++
	if b.state == HalfOpen || b.failures >= b.threshold {
		b.state, b.openedAt, b.trial = Open, b.now(), false
	}
}

// Release gives back a half-open trial whose outcome says nothing about the
// provider, such as a request the client cancelled.
func (b *Breaker) Release() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.trial = false
}

// State returns the current state without side effects.
func (b *Breaker) State() State {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.state
}
