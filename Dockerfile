FROM golang:1.25-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
# One package at a time: the Anthropic SDK needs a lot of memory to compile.
RUN CGO_ENABLED=0 go build -p 1 -trimpath -ldflags="-s -w" -o /out/ ./cmd/...

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/gateway /out/fakellm /usr/local/bin/
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/gateway"]
