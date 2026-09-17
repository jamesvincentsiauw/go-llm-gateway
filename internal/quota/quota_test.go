package quota

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
)

func newLimiter(t *testing.T) (*Limiter, *miniredis.Miniredis) {
	t.Helper()
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { _ = rdb.Close() })
	l := New(rdb)
	l.now = func() time.Time { return time.Date(2026, 9, 17, 23, 0, 0, 0, time.UTC) }
	return l, mr
}

func TestReserveWithinLimit(t *testing.T) {
	l, mr := newLimiter(t)
	ctx := context.Background()

	r, err := l.Reserve(ctx, "alpha", 300, 1000)
	if err != nil {
		t.Fatal(err)
	}
	if r.Remaining != 700 {
		t.Fatalf("remaining = %d, want 700", r.Remaining)
	}
	if got, _ := mr.Get("quota:alpha:2026-09-17"); got != "300" {
		t.Fatalf("stored = %q, want 300", got)
	}
	if ttl := mr.TTL("quota:alpha:2026-09-17"); ttl != keyTTL {
		t.Fatalf("ttl = %v, want %v", ttl, keyTTL)
	}
}

func TestReserveRejectsOverLimit(t *testing.T) {
	l, _ := newLimiter(t)
	ctx := context.Background()

	if _, err := l.Reserve(ctx, "alpha", 800, 1000); err != nil {
		t.Fatal(err)
	}
	_, err := l.Reserve(ctx, "alpha", 300, 1000)
	if !errors.Is(err, ErrExceeded) {
		t.Fatalf("err = %v, want ErrExceeded", err)
	}
	if _, err := l.Reserve(ctx, "beta", 300, 1000); err != nil {
		t.Fatalf("other clients have their own quota: %v", err)
	}
}

func TestSettleReturnsUnusedTokens(t *testing.T) {
	l, mr := newLimiter(t)
	ctx := context.Background()

	r, _ := l.Reserve(ctx, "alpha", 900, 1000)
	if err := r.Settle(ctx, 250); err != nil {
		t.Fatal(err)
	}
	if got, _ := mr.Get("quota:alpha:2026-09-17"); got != "250" {
		t.Fatalf("stored = %q, want 250", got)
	}
	if _, err := l.Reserve(ctx, "alpha", 700, 1000); err != nil {
		t.Fatalf("released tokens should be usable again: %v", err)
	}
}

func TestSettleCanChargeMoreThanReserved(t *testing.T) {
	l, mr := newLimiter(t)
	ctx := context.Background()

	r, _ := l.Reserve(ctx, "alpha", 100, 1000)
	_ = r.Settle(ctx, 140)
	_ = r.Settle(ctx, 140) // settling twice is a no-op
	if got, _ := mr.Get("quota:alpha:2026-09-17"); got != "140" {
		t.Fatalf("stored = %q, want 140", got)
	}
}

func TestSettleAfterExpiryDoesNotRecreateKey(t *testing.T) {
	l, mr := newLimiter(t)
	ctx := context.Background()

	r, _ := l.Reserve(ctx, "alpha", 100, 1000)
	mr.FastForward(keyTTL + time.Second)
	if err := r.Settle(ctx, 10); err != nil {
		t.Fatal(err)
	}
	if mr.Exists("quota:alpha:2026-09-17") {
		t.Fatal("settle recreated an expired key")
	}
}

func TestUnlimitedSkipsRedis(t *testing.T) {
	l, mr := newLimiter(t)
	r, err := l.Reserve(context.Background(), "alpha", 1_000_000, 0)
	if err != nil || r.Remaining != -1 {
		t.Fatalf("r = %+v, err = %v", r, err)
	}
	if err := r.Settle(context.Background(), 5); err != nil {
		t.Fatal(err)
	}
	if len(mr.Keys()) != 0 {
		t.Fatalf("keys = %v, want none", mr.Keys())
	}
}

func TestConcurrentReservationsNeverOvershoot(t *testing.T) {
	l, _ := newLimiter(t)
	var granted atomic.Int32
	var wg sync.WaitGroup
	for range 60 {
		wg.Go(func() {
			if _, err := l.Reserve(context.Background(), "alpha", 10, 200); err == nil {
				granted.Add(1)
			}
		})
	}
	wg.Wait()
	if granted.Load() != 20 {
		t.Fatalf("granted %d reservations of 10 under a limit of 200, want 20", granted.Load())
	}
}
