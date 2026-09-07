/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package schedulingdecisionlogger

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

// jsonlWriter appends one JSON object per line. Writes are serialized; each
// write is flushed so a process kill still leaves a complete trailing record.
type jsonlWriter struct {
	mu   sync.Mutex
	file *os.File
}

func openJSONL(path string) (*jsonlWriter, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open decision log %q: %w", path, err)
	}
	return &jsonlWriter{file: f}, nil
}

func (w *jsonlWriter) Write(record *DecisionRecord) error {
	if w == nil || w.file == nil {
		return nil
	}
	b, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("marshal decision record: %w", err)
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if _, err := w.file.Write(append(b, '\n')); err != nil {
		return fmt.Errorf("write decision record: %w", err)
	}
	return w.file.Sync()
}

func (w *jsonlWriter) Close() error {
	if w == nil || w.file == nil {
		return nil
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	err := w.file.Close()
	w.file = nil
	return err
}
