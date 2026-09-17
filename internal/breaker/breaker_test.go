package breaker

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type clock struct{ t time.Time }

func (c *clock) now() time.Time          { return c.t }
func (c *clock) advance(d time.Duration) { c.t = c.t.Add(d) }

func newTestBreaker() (*Breaker, *clock) {
	c := &clock{t: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)}
	b := New(3, 10*time.Second)
	b.now = c.now
	return b, c
}

func TestOpensAfterThreshold(t *testing.T) {
	b, _ := newTestBreaker()
	for range 2 {
		b.Failure()
	}
	if !b.Allow() {
		t.Fatal("breaker should stay closed below the threshold")
	}
	b.Failure()
	if b.State() != Open || b.Allow() {
		t.Fatalf("state = %v, want open and rejecting", b.State())
	}
}

func TestSuccessResetsFailureCount(t *testing.T) {
	b, _ := newTestBreaker()
	b.Failure()
	b.Failure()
	b.Success()
	b.Failure()
	if b.State() != Closed {
		t.Fatal("failures before a success should not count")
	}
}

func TestHalfOpenAllowsOneTrial(t *testing.T) {
	b, c := newTestBreaker()
	for range 3 {
		b.Failure()
	}
	c.advance(10 * time.Second)

	if !b.Allow() {
		t.Fatal("cooldown passed, the trial should be allowed")
	}
	if b.Allow() {
		t.Fatal("only one trial may run at a time")
	}
	b.Success()
	if b.State() != Closed || !b.Allow() {
		t.Fatal("a successful trial should close the breaker")
	}
}

func TestFailedTrialReopens(t *testing.T) {
	b, c := newTestBreaker()
	for range 3 {
		b.Failure()
	}
	c.advance(10 * time.Second)
	b.Allow()
	b.Failure()

	if b.State() != Open || b.Allow() {
		t.Fatal("a failed trial should reopen the breaker for a full cooldown")
	}
	c.advance(10 * time.Second)
	if !b.Allow() {
		t.Fatal("next trial should be allowed after another cooldown")
	}
}

func TestReleaseFreesTrial(t *testing.T) {
	b, c := newTestBreaker()
	for range 3 {
		b.Failure()
	}
	c.advance(10 * time.Second)
	b.Allow()
	b.Release()
	if !b.Allow() {
		t.Fatal("a released trial should let another request try")
	}
}

func TestConcurrentTrialIsExclusive(t *testing.T) {
	b, c := newTestBreaker()
	for range 3 {
		b.Failure()
	}
	c.advance(10 * time.Second)

	var allowed atomic.Int32
	var wg sync.WaitGroup
	for range 50 {
		wg.Go(func() {
			if b.Allow() {
				allowed.Add(1)
			}
		})
	}
	wg.Wait()
	if allowed.Load() != 1 {
		t.Fatalf("%d requests passed a half-open breaker, want 1", allowed.Load())
	}
}
