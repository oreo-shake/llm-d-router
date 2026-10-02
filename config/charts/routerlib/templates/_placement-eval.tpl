{{/*
Placement-eval (BLIS) sidecar helpers for simulated C^future / V_H.
*/}}

{{- define "llm-d-router.placementEval.env" -}}
{{- if .Values.router.placementEval.enabled }}
- name: PLACEMENT_EVAL_URL
  value: "http://localhost:{{ .Values.router.placementEval.port }}"
- name: PLACEMENT_EVAL_TIMEOUT_MS
  value: {{ .Values.router.placementEval.timeoutMs | quote }}
{{- range $key, $value := .Values.router.placementEval.eppEnv }}
- name: {{ $key }}
  value: {{ $value | quote }}
{{- end }}
{{- end }}
{{- end }}

{{- define "llm-d-router.placementEval.containers" -}}
{{- if .Values.router.placementEval.enabled }}
- name: placement-eval
  image: {{ .Values.router.placementEval.image.registry }}/{{ .Values.router.placementEval.image.repository }}:{{ .Values.router.placementEval.image.tag }}
  imagePullPolicy: {{ .Values.router.placementEval.image.pullPolicy }}
  args:
    - placement-eval-server
    - --listen
    - ":{{ .Values.router.placementEval.port }}"
  ports:
  - containerPort: {{ .Values.router.placementEval.port }}
    name: placement-eval
  livenessProbe:
    httpGet:
      path: /healthz
      port: {{ .Values.router.placementEval.port }}
    initialDelaySeconds: {{ .Values.router.placementEval.livenessProbe.initialDelaySeconds }}
    periodSeconds: {{ .Values.router.placementEval.livenessProbe.periodSeconds }}
  readinessProbe:
    httpGet:
      path: /readyz
      port: {{ .Values.router.placementEval.port }}
    initialDelaySeconds: {{ .Values.router.placementEval.readinessProbe.initialDelaySeconds }}
    periodSeconds: {{ .Values.router.placementEval.readinessProbe.periodSeconds }}
  resources:
    {{- toYaml .Values.router.placementEval.resources | nindent 4 }}
{{- end }}
{{- end }}
