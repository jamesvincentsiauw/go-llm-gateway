//go:build integration

// Package integration runs the gateway against a real Redis.
// Set REDIS_URL to use a server other than localhost:6379.
package integration

import (
	"crypto/rand"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/redis/go-redis/v9"

	"github.com/jamesvincentsiauw/go-llm-gateway/internal/api"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/breaker"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/failover"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/fakellm"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/metrics"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/provider"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/quota"
)

func redisClient(t *testing.T) *redis.Client {
	t.Helper()
	url := os.Getenv("REDIS_URL")
	if url == "" {
		url = "redis://localhost:6379/0"
	}
	opts, err := redis.ParseURL(url)
	if err != nil {
		t.Fatal(err)
	}
	rdb := redis.NewClient(opts)
	if err := rdb.Ping(t.Context()).Err(); err != nil {
		t.Fatalf("redis at %s: %v", url, err)
	}
	t.Cleanup(func() { _ = rdb.Close() })
	return rdb
}

func gateway(t *testing.T, rdb *redis.Client, client string, limit int) (url, key string) {
	t.Helper()
	upstream := httptest.NewServer(fakellm.Handler())
	t.Cleanup(upstream.Close)

	target := func(name, model string) failover.Target {
		return failover.Target{
			Provider: provider.NewOpenAI(name, upstream.URL+"/v1", "", nil),
			Model:    model,
			Breaker:  breaker.New(1000, time.Minute),
		}
	}
	m := metrics.New(prometheus.NewRegistry())
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	key = "gw-" + rand.Text()
	srv := &api.Server{
		Routes: map[string][]failover.Target{
			"flaky": {target("primary", "fake-cut-5"), target("backup", "fake")},
		},
		Clients:          map[[32]byte]api.Client{api.HashKey(key): {Name: client, DailyTokenLimit: limit}},
		Engine:           &failover.Engine{StallTimeout: 5 * time.Second, Observer: m, Log: log},
		Quota:            quota.New(rdb),
		DefaultMaxTokens: 200,
		MaxTokensLimit:   1000,
		Metrics:          m,
		Log:              log,
	}
	gw := httptest.NewServer(srv.Handler())
	t.Cleanup(gw.Close)
	return gw.URL, key
}

func TestConcurrentRequestsStayWithinQuota(t *testing.T) {
	rdb := redisClient(t)
	client := "it-" + rand.Text()[:8]
	dayKey := "quota:" + client + ":" + time.Now().UTC().Format("2006-01-02")
	t.Cleanup(func() { rdb.Del(t.Context(), dayKey) })

	const limit = 3000
	url, key := gateway(t, rdb, client, limit)

	var (
		mu       sync.Mutex
		used     int
		statuses = map[int]int{}
		wg       sync.WaitGroup
	)
	for range 40 {
		wg.Go(func() {
			req, _ := http.NewRequest(http.MethodPost, url+"/v1/chat/completions", strings.NewReader(
				`{"model":"flaky","messages":[{"role":"user","content":"tell me"}]}`))
			req.Header.Set("Authorization", "Bearer "+key)
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Error(err)
				return
			}
			defer resp.Body.Close()

			var body struct {
				Choices []struct {
					Message struct{ Content string } `json:"message"`
				} `json:"choices"`
				Usage struct {
					Total int `json:"total_tokens"`
				} `json:"usage"`
			}
			_ = json.NewDecoder(resp.Body).Decode(&body)

			mu.Lock()
			defer mu.Unlock()
			statuses[resp.StatusCode]++
			if resp.StatusCode == http.StatusOK {
				used += body.Usage.Total
				if body.Choices[0].Message.Content != fakellm.Passage {
					t.Errorf("stitched answer differs: %q", body.Choices[0].Message.Content)
				}
			}
		})
	}
	wg.Wait()

	if statuses[http.StatusOK] == 0 || statuses[http.StatusOK]+statuses[http.StatusTooManyRequests] != 40 {
		t.Fatalf("statuses = %v, want only 200 and 429", statuses)
	}
	stored, err := rdb.Get(t.Context(), dayKey).Result()
	if err != nil {
		t.Fatal(err)
	}
	if stored != strconv.Itoa(used) {
		t.Fatalf("redis holds %s tokens, successful responses reported %d", stored, used)
	}
	if used > limit {
		t.Fatalf("used %d tokens, above the limit of %d", used, limit)
	}
	if ttl := rdb.TTL(t.Context(), dayKey).Val(); ttl <= 0 {
		t.Fatalf("quota key has no expiry: %v", ttl)
	}
	t.Logf("statuses %v, %d tokens used of %d", statuses, used, limit)
}
