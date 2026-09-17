// Package config loads the gateway configuration file. Secrets are never stored
// in the file; entries name the environment variable that holds them.
package config

import (
	"errors"
	"fmt"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

// Provider types.
const (
	TypeAnthropic = "anthropic"
	TypeOpenAI    = "openai"
)

type Config struct {
	Listen           string        `yaml:"listen"`
	RedisURL         string        `yaml:"redis_url"`
	StallTimeout     time.Duration `yaml:"stall_timeout"`
	DefaultMaxTokens int           `yaml:"default_max_tokens"`
	MaxTokensLimit   int           `yaml:"max_tokens_limit"`
	Breaker          Breaker       `yaml:"breaker"`
	Providers        []Provider    `yaml:"providers"`
	Routes           []Route       `yaml:"routes"`
	Clients          []Client      `yaml:"clients"`
}

type Breaker struct {
	Failures int           `yaml:"failures"`
	Cooldown time.Duration `yaml:"cooldown"`
}

type Provider struct {
	Name             string `yaml:"name"`
	Type             string `yaml:"type"`
	BaseURL          string `yaml:"base_url"`
	APIKeyEnv        string `yaml:"api_key_env"`
	RefusalFallbacks bool   `yaml:"refusal_fallbacks"`

	APIKey string `yaml:"-"`
}

// Route maps the model name clients send to an ordered list of upstream models.
type Route struct {
	Model   string   `yaml:"model"`
	Targets []Target `yaml:"targets"`
}

type Target struct {
	Provider string `yaml:"provider"`
	Model    string `yaml:"model"`
}

type Client struct {
	Name            string `yaml:"name"`
	KeyEnv          string `yaml:"key_env"`
	DailyTokenLimit int    `yaml:"daily_token_limit"`

	Key string `yaml:"-"`
}

// Load reads path and resolves secrets from the environment.
func Load(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	return Parse(b, os.Getenv)
}

// Parse decodes b, applies defaults and validates the result.
func Parse(b []byte, getenv func(string) string) (*Config, error) {
	cfg := Config{
		Listen:           ":8080",
		RedisURL:         "redis://localhost:6379/0",
		StallTimeout:     20 * time.Second,
		DefaultMaxTokens: 4096,
		MaxTokensLimit:   64000,
		Breaker:          Breaker{Failures: 3, Cooldown: 30 * time.Second},
	}
	if err := yaml.Unmarshal(b, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	var errs []error
	providers := map[string]bool{}
	for i := range cfg.Providers {
		p := &cfg.Providers[i]
		switch {
		case p.Name == "":
			errs = append(errs, fmt.Errorf("providers[%d]: name is required", i))
		case providers[p.Name]:
			errs = append(errs, fmt.Errorf("provider %q is defined twice", p.Name))
		}
		providers[p.Name] = true
		switch p.Type {
		case TypeAnthropic:
		case TypeOpenAI:
			if p.BaseURL == "" {
				errs = append(errs, fmt.Errorf("provider %q: base_url is required for type openai", p.Name))
			}
		default:
			errs = append(errs, fmt.Errorf("provider %q: unknown type %q", p.Name, p.Type))
		}
		if p.APIKeyEnv != "" {
			p.APIKey = getenv(p.APIKeyEnv)
		}
	}

	routes := map[string]bool{}
	for _, r := range cfg.Routes {
		if r.Model == "" || routes[r.Model] {
			errs = append(errs, fmt.Errorf("route %q: model must be set and unique", r.Model))
		}
		routes[r.Model] = true
		if len(r.Targets) == 0 {
			errs = append(errs, fmt.Errorf("route %q: needs at least one target", r.Model))
		}
		for _, t := range r.Targets {
			if !providers[t.Provider] {
				errs = append(errs, fmt.Errorf("route %q: unknown provider %q", r.Model, t.Provider))
			}
			if t.Model == "" {
				errs = append(errs, fmt.Errorf("route %q: target on %q has no model", r.Model, t.Provider))
			}
		}
	}

	keys := map[string]string{}
	for i := range cfg.Clients {
		c := &cfg.Clients[i]
		c.Key = getenv(c.KeyEnv)
		switch {
		case c.Name == "":
			errs = append(errs, fmt.Errorf("clients[%d]: name is required", i))
		case c.Key == "":
			errs = append(errs, fmt.Errorf("client %q: environment variable %q is empty", c.Name, c.KeyEnv))
		case keys[c.Key] != "":
			errs = append(errs, fmt.Errorf("clients %q and %q share an API key", keys[c.Key], c.Name))
		}
		keys[c.Key] = c.Name
	}

	if len(cfg.Routes) == 0 {
		errs = append(errs, errors.New("at least one route is required"))
	}
	if len(cfg.Clients) == 0 {
		errs = append(errs, errors.New("at least one client is required"))
	}
	if cfg.DefaultMaxTokens < 1 || cfg.MaxTokensLimit < cfg.DefaultMaxTokens {
		errs = append(errs, errors.New("default_max_tokens must be positive and not above max_tokens_limit"))
	}
	if cfg.Breaker.Failures < 1 || cfg.Breaker.Cooldown <= 0 {
		errs = append(errs, errors.New("breaker.failures and breaker.cooldown must be positive"))
	}
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	return &cfg, nil
}
