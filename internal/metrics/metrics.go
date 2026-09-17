// Package metrics exposes gateway counters for Prometheus.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
)

type Metrics struct {
	Requests  *prometheus.CounterVec
	Attempts  *prometheus.CounterVec
	Failovers *prometheus.CounterVec
	Tokens    *prometheus.CounterVec
	Duration  *prometheus.HistogramVec
}

func New(reg prometheus.Registerer) *Metrics {
	m := &Metrics{
		Requests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_requests_total",
			Help: "Chat completion requests, by route and HTTP status.",
		}, []string{"route", "status"}),
		Attempts: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_upstream_attempts_total",
			Help: "Upstream attempts, by provider and outcome.",
		}, []string{"provider", "outcome"}),
		Failovers: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_failovers_total",
			Help: "Switches between upstreams. kind is before_output or mid_stream.",
		}, []string{"from", "to", "kind"}),
		Tokens: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_tokens_total",
			Help: "Tokens reported by upstreams, by client and direction.",
		}, []string{"client", "direction"}),
		Duration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "gateway_request_duration_seconds",
			Help:    "Time from request to the last byte of the response.",
			Buckets: []float64{0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120},
		}, []string{"route"}),
	}
	reg.MustRegister(m.Requests, m.Attempts, m.Failovers, m.Tokens, m.Duration)
	return m
}

// Attempt implements failover.Observer.
func (m *Metrics) Attempt(provider, outcome string) {
	m.Attempts.WithLabelValues(provider, outcome).Inc()
}

// FailedOver implements failover.Observer.
func (m *Metrics) FailedOver(from, to string, midStream bool) {
	kind := "before_output"
	if midStream {
		kind = "mid_stream"
	}
	m.Failovers.WithLabelValues(from, to, kind).Inc()
}
