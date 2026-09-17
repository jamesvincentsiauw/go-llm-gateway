// Command gateway runs the LLM gateway.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"

	"github.com/jamesvincentsiauw/go-llm-gateway/internal/api"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/breaker"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/config"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/failover"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/metrics"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/provider"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/quota"
)

func main() {
	configPath := flag.String("config", "gateway.yaml", "path to the configuration file")
	flag.Parse()

	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	if err := run(*configPath, log); err != nil {
		log.Error("gateway stopped", "err", err)
		os.Exit(1)
	}
}

func run(configPath string, log *slog.Logger) error {
	cfg, err := config.Load(configPath)
	if err != nil {
		return err
	}

	redisOpts, err := redis.ParseURL(cfg.RedisURL)
	if err != nil {
		return fmt.Errorf("redis_url: %w", err)
	}
	rdb := redis.NewClient(redisOpts)
	defer rdb.Close()

	reg := prometheus.NewRegistry()
	reg.MustRegister(collectors.NewGoCollector(), collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))
	m := metrics.New(reg)

	srv := &api.Server{
		Routes:           buildRoutes(cfg, reg),
		Clients:          map[[32]byte]api.Client{},
		Engine:           &failover.Engine{StallTimeout: cfg.StallTimeout, Observer: m, Log: log},
		Quota:            quota.New(rdb),
		DefaultMaxTokens: cfg.DefaultMaxTokens,
		MaxTokensLimit:   cfg.MaxTokensLimit,
		Metrics:          m,
		Log:              log,
	}
	for _, c := range cfg.Clients {
		srv.Clients[api.HashKey(c.Key)] = api.Client{Name: c.Name, DailyTokenLimit: c.DailyTokenLimit}
	}

	mux := http.NewServeMux()
	mux.Handle("/v1/", srv.Handler())
	mux.Handle("GET /metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		if err := rdb.Ping(r.Context()).Err(); err != nil {
			http.Error(w, "redis unavailable", http.StatusServiceUnavailable)
			return
		}
		_, _ = w.Write([]byte("ok\n"))
	})

	httpSrv := &http.Server{Addr: cfg.Listen, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errc := make(chan error, 1)
	go func() {
		log.Info("listening", "addr", cfg.Listen, "routes", len(cfg.Routes), "clients", len(cfg.Clients))
		errc <- httpSrv.ListenAndServe()
	}()

	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
	}

	// Streams can run for minutes, so give in-flight answers time to finish.
	log.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	if err := httpSrv.Shutdown(shutdownCtx); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

func buildRoutes(cfg *config.Config, reg prometheus.Registerer) map[string][]failover.Target {
	providers := map[string]provider.Provider{}
	breakers := map[string]*breaker.Breaker{}
	streamClient := &http.Client{Transport: http.DefaultTransport}

	for _, p := range cfg.Providers {
		switch p.Type {
		case config.TypeAnthropic:
			providers[p.Name] = provider.NewAnthropic(p.Name, provider.AnthropicOptions{
				APIKey:           p.APIKey,
				BaseURL:          p.BaseURL,
				RefusalFallbacks: p.RefusalFallbacks,
			})
		case config.TypeOpenAI:
			providers[p.Name] = provider.NewOpenAI(p.Name, p.BaseURL, p.APIKey, streamClient)
		}

		b := breaker.New(cfg.Breaker.Failures, cfg.Breaker.Cooldown)
		breakers[p.Name] = b
		reg.MustRegister(prometheus.NewGaugeFunc(prometheus.GaugeOpts{
			Name:        "gateway_breaker_state",
			Help:        "Circuit breaker state per provider: 0 closed, 1 open, 2 half-open.",
			ConstLabels: prometheus.Labels{"provider": p.Name},
		}, func() float64 { return float64(b.State()) }))
	}

	routes := map[string][]failover.Target{}
	for _, r := range cfg.Routes {
		for _, t := range r.Targets {
			routes[r.Model] = append(routes[r.Model], failover.Target{
				Provider: providers[t.Provider],
				Model:    t.Model,
				Breaker:  breakers[t.Provider],
			})
		}
	}
	return routes
}
