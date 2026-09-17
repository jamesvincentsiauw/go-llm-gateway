package provider

import (
	"context"
	"errors"
	"io"
	"net/http"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
)

// Anthropic streams from the Claude Messages API through the official SDK.
type Anthropic struct {
	name             string
	client           anthropic.Client
	refusalFallbacks bool
}

// AnthropicOptions configures an Anthropic provider.
type AnthropicOptions struct {
	APIKey  string
	BaseURL string // optional, for tests or a proxy
	// RefusalFallbacks lets the API re-run a request that a model's safety
	// classifier declined on Anthropic's recommended fallback model.
	RefusalFallbacks bool
	HTTPClient       *http.Client
}

// NewAnthropic returns a Claude provider. SDK retries are disabled because the
// gateway decides whether to retry or move to another provider.
func NewAnthropic(name string, opts AnthropicOptions) *Anthropic {
	reqOpts := []option.RequestOption{option.WithMaxRetries(0)}
	if opts.APIKey != "" {
		reqOpts = append(reqOpts, option.WithAPIKey(opts.APIKey))
	}
	if opts.BaseURL != "" {
		reqOpts = append(reqOpts, option.WithBaseURL(opts.BaseURL))
	}
	if opts.HTTPClient != nil {
		reqOpts = append(reqOpts, option.WithHTTPClient(opts.HTTPClient))
	}
	return &Anthropic{name: name, client: anthropic.NewClient(reqOpts...), refusalFallbacks: opts.RefusalFallbacks}
}

func (p *Anthropic) Name() string { return p.name }

// Stream opens a streaming request. Temperature and top_p are not forwarded:
// current Claude models reject sampling parameters with a 400.
func (p *Anthropic) Stream(ctx context.Context, req Request) (Stream, error) {
	params := anthropic.BetaMessageNewParams{
		Model:     req.Model,
		MaxTokens: int64(req.MaxTokens),
	}
	for _, m := range req.Messages {
		switch m.Role {
		case "system":
			params.System = append(params.System, anthropic.BetaTextBlockParam{Text: m.Content})
		case "assistant":
			params.Messages = append(params.Messages, anthropic.BetaMessageParam{
				Role:    anthropic.BetaMessageParamRoleAssistant,
				Content: []anthropic.BetaContentBlockParamUnion{anthropic.NewBetaTextBlock(m.Content)},
			})
		default:
			params.Messages = append(params.Messages, anthropic.NewBetaUserMessage(anthropic.NewBetaTextBlock(m.Content)))
		}
	}
	if p.refusalFallbacks {
		params.Betas = []anthropic.AnthropicBeta{anthropic.AnthropicBetaServerSideFallback2026_07_01}
		params.Fallbacks = anthropic.BetaFallbacksParamUnion{OfDefault: constant.ValueOf[constant.Default]()}
	}

	return &anthropicStream{name: p.name, ctx: ctx, stream: p.client.Beta.Messages.NewStreaming(ctx, params)}, nil
}

type anthropicStream struct {
	name    string
	ctx     context.Context
	stream  *ssestream.Stream[anthropic.BetaRawMessageStreamEventUnion]
	chunk   Chunk
	err     error
	input   int
	stopped bool
}

func (s *anthropicStream) Next() bool {
	for s.stream.Next() {
		ev := s.stream.Current()
		switch ev.Type {
		case "message_start":
			u := ev.AsMessageStart().Message.Usage
			s.input = int(u.InputTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens)
		case "content_block_delta":
			if d, ok := ev.AsContentBlockDelta().Delta.AsAny().(anthropic.BetaTextDelta); ok && d.Text != "" {
				s.chunk = Chunk{Text: d.Text}
				return true
			}
		case "message_delta":
			md := ev.AsMessageDelta()
			input := int(md.Usage.InputTokens + md.Usage.CacheCreationInputTokens + md.Usage.CacheReadInputTokens)
			if input == 0 {
				input = s.input
			}
			s.chunk = Chunk{
				FinishReason: finishReason(md.Delta.StopReason),
				Usage:        &Usage{InputTokens: input, OutputTokens: int(md.Usage.OutputTokens)},
			}
			return true
		case "message_stop":
			s.stopped = true
		}
	}

	err := s.stream.Err()
	if err == nil {
		if s.stopped {
			return false
		}
		// The body ended before message_stop, so the upstream went away mid-answer.
		err = io.ErrUnexpectedEOF
	}
	var apierr *anthropic.Error
	switch {
	case errors.As(err, &apierr):
		s.err = &Error{Provider: s.name, Status: apierr.StatusCode, Retryable: RetryableStatus(apierr.StatusCode), Err: err}
	case s.ctx.Err() != nil:
		s.err = &Error{Provider: s.name, Err: s.ctx.Err()}
	default:
		if errors.Is(err, io.EOF) {
			err = io.ErrUnexpectedEOF
		}
		s.err = &Error{Provider: s.name, Retryable: true, Err: err}
	}
	return false
}

func finishReason(r anthropic.BetaStopReason) string {
	switch r {
	case anthropic.BetaStopReasonMaxTokens:
		return FinishLength
	case anthropic.BetaStopReasonRefusal:
		return FinishContentFilter
	default:
		return FinishStop
	}
}

func (s *anthropicStream) Chunk() Chunk { return s.chunk }
func (s *anthropicStream) Err() error   { return s.err }
func (s *anthropicStream) Close() error { return s.stream.Close() }
