package config

import (
	"strings"
	"testing"
	"time"
)

const valid = `
stall_timeout: 5s
providers:
  - name: claude
    type: anthropic
    api_key_env: ANTHROPIC_API_KEY
    refusal_fallbacks: true
  - name: local
    type: openai
    base_url: http://fake:9000/v1
routes:
  - model: smart
    targets:
      - {provider: claude, model: claude-opus-5}
      - {provider: local, model: fake}
clients:
  - name: alpha
    key_env: KEY_ALPHA
    daily_token_limit: 1000
`

func env(m map[string]string) func(string) string {
	return func(k string) string { return m[k] }
}

func TestParseValid(t *testing.T) {
	cfg, err := Parse([]byte(valid), env(map[string]string{"ANTHROPIC_API_KEY": "sk-ant", "KEY_ALPHA": "gw-alpha"}))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.StallTimeout != 5*time.Second || cfg.Listen != ":8080" || cfg.Breaker.Failures != 3 {
		t.Fatalf("defaults or overrides not applied: %+v", cfg)
	}
	if cfg.Providers[0].APIKey != "sk-ant" || !cfg.Providers[0].RefusalFallbacks {
		t.Fatalf("provider = %+v", cfg.Providers[0])
	}
	if cfg.Clients[0].Key != "gw-alpha" || cfg.Routes[0].Targets[1].Model != "fake" {
		t.Fatalf("clients = %+v, routes = %+v", cfg.Clients, cfg.Routes)
	}
}

func TestParseReportsEveryProblem(t *testing.T) {
	bad := `
default_max_tokens: 0
providers:
  - name: p
    type: grpc
  - name: p
    type: openai
routes:
  - model: smart
    targets:
      - {provider: missing, model: x}
      - {provider: p}
clients:
  - name: alpha
    key_env: KEY_ALPHA
  - name: beta
    key_env: KEY_BETA
`
	_, err := Parse([]byte(bad), env(map[string]string{"KEY_BETA": "same", "KEY_ALPHA": "same"}))
	if err == nil {
		t.Fatal("expected errors")
	}
	for _, want := range []string{
		`unknown type "grpc"`,
		`provider "p" is defined twice`,
		"base_url is required",
		`unknown provider "missing"`,
		"has no model",
		"share an API key",
		"default_max_tokens",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error does not mention %q:\n%v", want, err)
		}
	}
}

func TestParseRequiresClientKey(t *testing.T) {
	_, err := Parse([]byte(valid), env(nil))
	if err == nil || !strings.Contains(err.Error(), `"KEY_ALPHA" is empty`) {
		t.Fatalf("err = %v", err)
	}
}
