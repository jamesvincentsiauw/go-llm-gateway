package failover

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jamesvincentsiauw/go-llm-gateway/internal/breaker"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/fakellm"
	"github.com/jamesvincentsiauw/go-llm-gateway/internal/provider"
)

// scripted replays chunks, then fails with err (if set).
type scripted struct {
	name    string
	chunks  []provider.Chunk
	err     error
	openErr error
	delay   time.Duration

	mu   sync.Mutex
	reqs []provider.Request
}

func (s *scripted) Name() string { return s.name }

func (s *scripted) Stream(ctx context.Context, req provider.Request) (provider.Stream, error) {
	s.mu.Lock()
	s.reqs = append(s.reqs, req)
	s.mu.Unlock()
	if s.openErr != nil {
		return nil, s.openErr
	}
	return &scriptedStream{ctx: ctx, p: s, i: -1}, nil
}

type scriptedStream struct {
	ctx context.Context
	p   *scripted
	i   int
	err error
}

func (s *scriptedStream) Next() bool {
	if s.p.delay > 0 {
		select {
		case <-time.After(s.p.delay):
		case <-s.ctx.Done():
			s.err = &provider.Error{Provider: s.p.name, Err: s.ctx.Err()}
			return false
		}
	}
	s.i++
	if s.i < len(s.p.chunks) {
		return true
	}
	s.err = s.p.err
	return false
}

func (s *scriptedStream) Chunk() provider.Chunk { return s.p.chunks[s.i] }
func (s *scriptedStream) Err() error            { return s.err }
func (s *scriptedStream) Close() error          { return nil }

type recorder struct {
	text      strings.Builder
	failovers []string
	attempts  []string
	failText  error
}

func (r *recorder) Text(s string) error {
	if r.failText != nil {
		return r.failText
	}
	r.text.WriteString(s)
	return nil
}

func (r *recorder) Failover(from, to string, mid bool) error {
	kind := "before-output"
	if mid {
		kind = "mid-stream"
	}
	r.failovers = append(r.failovers, from+">"+to+":"+kind)
	return nil
}

func (r *recorder) Attempt(p, outcome string)            { r.attempts = append(r.attempts, p+":"+outcome) }
func (r *recorder) FailedOver(from, to string, mid bool) {}

func texts(parts ...string) []provider.Chunk {
	var out []provider.Chunk
	for _, p := range parts {
		out = append(out, provider.Chunk{Text: p})
	}
	return out
}

func done(parts ...string) []provider.Chunk {
	return append(texts(parts...), provider.Chunk{FinishReason: provider.FinishStop, Usage: &provider.Usage{InputTokens: 10, OutputTokens: 5}})
}

func target(p provider.Provider) Target {
	return Target{Provider: p, Model: "m-" + p.Name(), Breaker: breaker.New(1, time.Hour)}
}

func newEngine(rec *recorder) *Engine {
	return &Engine{Observer: rec, Log: slog.New(slog.NewTextHandler(io.Discard, nil))}
}

var dropped = &provider.Error{Provider: "x", Retryable: true, Err: io.ErrUnexpectedEOF}

func baseRequest() provider.Request {
	return provider.Request{MaxTokens: 1000, Messages: []provider.Message{{Role: "user", Content: "explain"}}}
}

func TestSingleUpstreamSuccess(t *testing.T) {
	rec := &recorder{}
	a := &scripted{name: "a", chunks: done("Hello", " wor", "ld", "!")}

	res, err := newEngine(rec).Run(context.Background(), baseRequest(), []Target{target(a)}, rec)
	if err != nil {
		t.Fatal(err)
	}
	if rec.text.String() != "Hello world!" || res.FinishReason != provider.FinishStop {
		t.Fatalf("text = %q, finish = %q", rec.text.String(), res.FinishReason)
	}
	if res.Usage.OutputTokens != 5 || len(res.Providers) != 1 || a.reqs[0].Model != "m-a" {
		t.Fatalf("result = %+v, request = %+v", res, a.reqs[0])
	}
}

func TestFailoverBeforeOutputResendsOriginalRequest(t *testing.T) {
	rec := &recorder{}
	a := &scripted{name: "a", openErr: &provider.Error{Status: 503, Retryable: true, Err: errors.New("busy")}}
	b := &scripted{name: "b", chunks: done("fine")}

	res, err := newEngine(rec).Run(context.Background(), baseRequest(), []Target{target(a), target(b)}, rec)
	if err != nil {
		t.Fatal(err)
	}
	if rec.text.String() != "fine" || len(b.reqs[0].Messages) != 1 {
		t.Fatalf("text = %q, second request = %+v", rec.text.String(), b.reqs[0])
	}
	if strings.Join(rec.failovers, ",") != "a>b:before-output" || strings.Join(res.Providers, ",") != "b" {
		t.Fatalf("failovers = %v, providers = %v", rec.failovers, res.Providers)
	}
}

func TestMidStreamFailoverContinuesFromLastWord(t *testing.T) {
	rec := &recorder{}
	a := &scripted{name: "a", chunks: texts("The answer", " is for", "ty"), err: dropped}
	b := &scripted{name: "b", chunks: done(" forty-two.")}

	res, err := newEngine(rec).Run(context.Background(), baseRequest(), []Target{target(a), target(b)}, rec)
	if err != nil {
		t.Fatal(err)
	}
	// "forty" was held back when a dropped, so b rewrote it.
	if got := rec.text.String(); got != "The answer is forty-two." {
		t.Fatalf("text = %q", got)
	}

	cont := b.reqs[0]
	if len(cont.Messages) != 3 || cont.Messages[1].Role != "assistant" || cont.Messages[1].Content != "The answer is" {
		t.Fatalf("continuation messages = %+v", cont.Messages)
	}
	if cont.Messages[2].Content != ContinuationPrompt {
		t.Fatalf("last message = %q", cont.Messages[2].Content)
	}
	if cont.MaxTokens != 1000-EstimateTokens("The answer is") {
		t.Fatalf("max tokens = %d", cont.MaxTokens)
	}
	if strings.Join(rec.failovers, ",") != "a>b:mid-stream" || strings.Join(res.Providers, ",") != "a,b" {
		t.Fatalf("failovers = %v, providers = %v", rec.failovers, res.Providers)
	}
}

func TestContinuationGetsSpaceWhenModelOmitsIt(t *testing.T) {
	rec := &recorder{}
	a := &scripted{name: "a", chunks: texts("One two", " thr"), err: dropped}
	b := &scripted{name: "b", chunks: done("three four")}

	if _, err := newEngine(rec).Run(context.Background(), baseRequest(), []Target{target(a), target(b)}, rec); err != nil {
		t.Fatal(err)
	}
	if got := rec.text.String(); got != "One two three four" {
		t.Fatalf("text = %q", got)
	}
}

func TestDropBeforeFirstWordIsNotMidStream(t *testing.T) {
	rec := &recorder{}
	a := &scripted{name: "a", chunks: texts("Hel"), err: dropped}
	b := &scripted{name: "b", chunks: done("Hello there")}

	if _, err := newEngine(rec).Run(context.Background(), baseRequest(), []Target{target(a), target(b)}, rec); err != nil {
		t.Fatal(err)
	}
	if rec.text.String() != "Hello there" || len(b.reqs[0].Messages) != 1 {
		t.Fatalf("text = %q, b messages = %d", rec.text.String(), len(b.reqs[0].Messages))
	}
	if rec.failovers[0] != "a>b:before-output" {
		t.Fatalf("failover = %v", rec.failovers)
	}
}

func TestNonRetryableErrorStops(t *testing.T) {
	rec := &recorder{}
	a := &scripted{name: "a", openErr: &provider.Error{Status: 400, Err: errors.New("bad request")}}
	b := &scripted{name: "b", chunks: done("unused")}
	ta := target(a)

	_, err := newEngine(rec).Run(context.Background(), baseRequest(), []Target{ta, target(b)}, rec)
	var perr *provider.Error
	if !errors.As(err, &perr) || perr.Status != 400 {
		t.Fatalf("err = %v, want the 400", err)
	}
	if len(b.reqs) != 0 {
		t.Fatal("a rejected request must not be sent to the next upstream")
	}
	if ta.Breaker.State() != breaker.Closed {
		t.Fatal("a client error must not trip the breaker")
	}
}

func TestAllUpstreamsFail(t *testing.T) {
	rec := &recorder{}
	a := &scripted{name: "a", openErr: dropped}
	b := &scripted{name: "b", openErr: dropped}
	ta, tb := target(a), target(b)

	_, err := newEngine(rec).Run(context.Background(), baseRequest(), []Target{ta, tb}, rec)
	if !errors.Is(err, ErrExhausted) {
		t.Fatalf("err = %v, want ErrExhausted", err)
	}
	if ta.Breaker.State() != breaker.Open || tb.Breaker.State() != breaker.Open {
		t.Fatal("both breakers should open after their failure")
	}

	// With both breakers open the next request is skipped entirely.
	rec2 := &recorder{}
	_, err = newEngine(rec2).Run(context.Background(), baseRequest(), []Target{ta, tb}, rec2)
	if !errors.Is(err, ErrExhausted) || strings.Join(rec2.attempts, ",") != "a:breaker_open,b:breaker_open" {
		t.Fatalf("err = %v, attempts = %v", err, rec2.attempts)
	}
	if len(a.reqs) != 1 {
		t.Fatal("an open breaker must not send the request")
	}
}

func TestStalledUpstreamFailsOver(t *testing.T) {
	rec := &recorder{}
	a := &scripted{name: "a", chunks: done("never"), delay: time.Second}
	b := &scripted{name: "b", chunks: done("quick")}
	e := newEngine(rec)
	e.StallTimeout = 50 * time.Millisecond

	start := time.Now()
	if _, err := e.Run(context.Background(), baseRequest(), []Target{target(a), target(b)}, rec); err != nil {
		t.Fatal(err)
	}
	if rec.text.String() != "quick" || time.Since(start) > 500*time.Millisecond {
		t.Fatalf("text = %q after %v", rec.text.String(), time.Since(start))
	}
	if rec.attempts[0] != "a:stalled" {
		t.Fatalf("attempts = %v", rec.attempts)
	}
}

func TestClientDisconnectStopsWithoutFailover(t *testing.T) {
	rec := &recorder{failText: errors.New("broken pipe")}
	a := &scripted{name: "a", chunks: done("Hello there friend")}
	b := &scripted{name: "b", chunks: done("unused")}
	ta := target(a)

	_, err := newEngine(rec).Run(context.Background(), baseRequest(), []Target{ta, target(b)}, rec)
	if err == nil || len(b.reqs) != 0 {
		t.Fatalf("err = %v, b requests = %d", err, len(b.reqs))
	}
	if ta.Breaker.State() != breaker.Closed {
		t.Fatal("a client disconnect must not trip the breaker")
	}
}

func TestSpentBudgetEndsWithLength(t *testing.T) {
	rec := &recorder{}
	a := &scripted{name: "a", chunks: texts("aaaa bbbb cccc", " dd"), err: dropped}
	b := &scripted{name: "b", chunks: done("unused")}
	req := baseRequest()
	req.MaxTokens = 3

	res, err := newEngine(rec).Run(context.Background(), req, []Target{target(a), target(b)}, rec)
	if err != nil {
		t.Fatal(err)
	}
	if res.FinishReason != provider.FinishLength || len(b.reqs) != 0 {
		t.Fatalf("finish = %q, b requests = %d", res.FinishReason, len(b.reqs))
	}
}

// TestStitchedAnswerMatchesUninterruptedOne runs the real OpenAI-compatible
// provider against the fake server, cutting the first upstream mid-answer.
func TestStitchedAnswerMatchesUninterruptedOne(t *testing.T) {
	srv := httptest.NewServer(fakellm.Handler())
	defer srv.Close()
	p1 := provider.NewOpenAI("primary", srv.URL+"/v1", "", srv.Client())
	p2 := provider.NewOpenAI("secondary", srv.URL+"/v1", "", srv.Client())

	for _, cut := range []int{1, 7, 20} {
		rec := &recorder{}
		targets := []Target{
			{Provider: p1, Model: fmt.Sprintf("fake-cut-%d", cut), Breaker: breaker.New(5, time.Hour)},
			{Provider: p2, Model: "fake", Breaker: breaker.New(5, time.Hour)},
		}
		res, err := newEngine(rec).Run(context.Background(), baseRequest(), targets, rec)
		if err != nil {
			t.Fatalf("cut %d: %v", cut, err)
		}
		if got := rec.text.String(); got != fakellm.Passage {
			t.Fatalf("cut %d: stitched text differs\n got: %q\nwant: %q", cut, got, fakellm.Passage)
		}
		if res.FinishReason != provider.FinishStop {
			t.Fatalf("cut %d: finish = %q", cut, res.FinishReason)
		}
	}
}
