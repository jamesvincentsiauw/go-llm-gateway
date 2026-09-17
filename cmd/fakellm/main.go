// Command fakellm serves the scripted OpenAI-compatible model used in demos.
package main

import (
	"flag"
	"log"
	"net/http"
	"time"

	"github.com/jamesvincentsiauw/go-llm-gateway/internal/fakellm"
)

func main() {
	addr := flag.String("addr", ":9000", "listen address")
	flag.Parse()

	srv := &http.Server{Addr: *addr, Handler: fakellm.Handler(), ReadHeaderTimeout: 5 * time.Second}
	log.Printf("fakellm listening on %s", *addr)
	log.Fatal(srv.ListenAndServe())
}
