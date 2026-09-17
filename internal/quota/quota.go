// Package quota enforces per-client daily token limits in Redis. Requests reserve
// their worst case up front, so concurrent requests cannot overshoot the limit.
package quota

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

// ErrExceeded is returned when a reservation does not fit in today's limit.
var ErrExceeded = errors.New("daily token quota exceeded")

const keyTTL = 48 * time.Hour

var reserveScript = redis.NewScript(`
local used = tonumber(redis.call('GET', KEYS[1]) or '0')
local n, limit = tonumber(ARGV[1]), tonumber(ARGV[2])
if used + n > limit then
  return {0, limit - used}
end
local total = redis.call('INCRBY', KEYS[1], n)
if redis.call('TTL', KEYS[1]) == -1 then
  redis.call('EXPIRE', KEYS[1], ARGV[3])
end
return {1, limit - total}
`)

// settleScript only touches keys that still exist, so a late settle never
// recreates an expired day without a TTL.
var settleScript = redis.NewScript(`
if redis.call('EXISTS', KEYS[1]) == 0 then
  return 0
end
return redis.call('INCRBY', KEYS[1], ARGV[1])
`)

// Limiter reserves tokens against daily limits.
type Limiter struct {
	rdb redis.Scripter
	now func() time.Time
}

// New returns a Limiter backed by rdb.
func New(rdb redis.Scripter) *Limiter {
	return &Limiter{rdb: rdb, now: time.Now}
}

// Reservation is a hold on tokens that must be settled once usage is known.
type Reservation struct {
	limiter   *Limiter
	key       string
	reserved  int
	Remaining int // tokens left today after this reservation, -1 when unlimited
}

// Reserve holds n tokens for client. A limit of zero or less means unlimited.
func (l *Limiter) Reserve(ctx context.Context, client string, n, limit int) (*Reservation, error) {
	if limit <= 0 {
		return &Reservation{Remaining: -1}, nil
	}
	key := fmt.Sprintf("quota:%s:%s", client, l.now().UTC().Format("2006-01-02"))
	res, err := reserveScript.Run(ctx, l.rdb, []string{key}, n, limit, int(keyTTL.Seconds())).Int64Slice()
	if err != nil {
		return nil, fmt.Errorf("reserve quota: %w", err)
	}
	if res[0] == 0 {
		return nil, fmt.Errorf("%w: %d tokens left, request needs up to %d", ErrExceeded, max(res[1], 0), n)
	}
	return &Reservation{limiter: l, key: key, reserved: n, Remaining: int(res[1])}, nil
}

// Settle replaces the reserved amount with actual usage. Actual usage can be
// higher than the reservation when the input estimate was low.
func (r *Reservation) Settle(ctx context.Context, actual int) error {
	if r.limiter == nil {
		return nil
	}
	delta := actual - r.reserved
	r.reserved = actual
	if delta == 0 {
		return nil
	}
	if err := settleScript.Run(ctx, r.limiter.rdb, []string{r.key}, delta).Err(); err != nil {
		return fmt.Errorf("settle quota: %w", err)
	}
	return nil
}
