.PHONY: up down logs build test test-integration lint demo

KEY ?= demo-key
URL ?= http://localhost:8080

up:
	docker compose build gateway
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f gateway

build:
	go build -o bin/ ./cmd/...

test:
	go test -race ./...

test-integration:
	go test -race -tags integration -count=1 ./integration/...

lint:
	golangci-lint run --build-tags integration ./...

# Stream the flaky route: the first upstream drops mid-answer and the backup finishes it.
demo:
	curl -sN $(URL)/v1/chat/completions -H "Authorization: Bearer $(KEY)" \
	  -H "Content-Type: application/json" \
	  -d '{"model":"demo-flaky","stream":true,"messages":[{"role":"user","content":"Why fail over mid-stream?"}]}'
