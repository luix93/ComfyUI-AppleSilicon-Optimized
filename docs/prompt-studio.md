# Apple Silicon Prompt Studio

A prompt-writing workspace inside ComfyUI. It produces **two STRING outputs**:
`prompt` and `negative_prompt`. It does not load diffusion models or generate
images, videos or audio itself.

## Start

1. Restart ComfyUI after installation and refresh the browser. Add **Apple
   Silicon → Prompting → Apple Silicon Prompt Studio**, or open the
   [example workflow](../workflows/apple_silicon_prompt_studio.json).
2. Select **Open Studio → Connections**. Start your LLM server, enter its API
   base URL and optional API key, then **Save & discover models**. Choose a model
   and **Save connection**. You can also type a model ID directly.
3. In **Compose**, choose the target model profile and describe your idea. Adjust
   style, detail, aspect ratio, duration and extra direction as appropriate.
4. **Generate prompt**, edit the result if needed, and connect the `prompt`
   output to a workflow text input. For CLIP Text Encode, convert its text widget
   to an input from its context menu. Connect `negative_prompt` separately where
   the target workflow uses negative conditioning.

Connect an image loader to **images**, or a video loader to **video** (native
VIDEO) or **images** (frame batches). Write your brief and press **Generate
prompt**. The references and brief go to the LLM together; no separate queue
action or capability checkbox is needed. You can also use **+ Attach reference**
directly on the node.

The canvas node contains the brief, profile, **LLM model** picker, editable
result, Generate button and **Generate on queue** toggle. Connection setup stays
in Open Studio. Discovered model lists are saved per connection and remain
available after restarting ComfyUI, refreshing the browser, or restoring a
workspace backup. Use **↻** beside the model picker to refresh the list; failed
discovery keeps the last saved list. Changing an endpoint clears its cached list.
The selected model is stored with this node's workflow settings, so separate
nodes can use different models without changing the connection's default.

The larger workspace has Compose, Library, Profiles,
Connections and Workspace sections. Both classic rendering and Nodes 2.0 use
the same DOM controls; hidden backing widgets retain the workflow values.

## Connections

| Server | Usual API base URL | Notes |
|---|---|---|
| oMLX | `http://localhost:8000/v1` | Enter the API key if your server requires one. |
| LM Studio | `http://localhost:1234/v1` | Enable its local server and make a model available. |
| Ollama | `http://localhost:11434/v1` | Uses Ollama's OpenAI compatibility API. |
| Other OpenAI-compatible server | Your server's API base | Must expose `/models` and `/chat/completions`; model discovery is optional if you enter an ID. |

Requests originate on the **ComfyUI machine**. `localhost` refers to that machine,
not a remote browser. A root URL is normalized to `/v1`; explicit proxy paths are
preserved. Redirects are rejected rather than forwarding credentials elsewhere.

The node sends ordinary non-streaming chat requests with `max_tokens` and
`temperature`, without requiring server-specific structured-output extensions.
It asks for JSON containing positive and negative text, and also accepts plain
text from simpler models. Empty, truncated, or malformed JSON answers produce
an actionable error rather than silently sending a partial prompt downstream.
For reasoning models, increase **Max output tokens** or disable thinking in the
server if they run out of space before answering. A nonnegative seed is sent
when requested; determinism depends on the server.

**API keys** are stored separately in a file created with owner-only `0600`
permissions. They are not included in graph widgets, library entries or exported
workspace JSON. Alternatively, specify an environment variable available to the
ComfyUI process. Environment variables take precedence over saved keys. Changing
an endpoint clears its saved key unless a replacement is explicitly entered.

## Drafts and workflow execution

- **Generate on queue: off** is the default. It returns the current positive and
  negative text unchanged, including whitespace, and makes no LLM request.
- **Generate on queue: on** requests a fresh prompt on each execution and updates
  the displayed result. It does not rely on ComfyUI's cached node output.
  Existing workflows retain their saved execution mode.
- **Generate prompt** includes connected inputs automatically. It submits just
  this node and its upstream dependencies to ComfyUI; downstream and unrelated
  nodes are excluded. Existing upstream outputs can be reused from ComfyUI's
  cache. Each click makes a fresh LLM request and leaves the workflow's execution
  mode unchanged. If a connected input requires generation, those upstream nodes
  must run first. If ComfyUI is busy, the request waits in its queue.
- Cancellation targets this generation's job. Whether the LLM immediately stops
  computing after its connection closes depends on the server. Upstream errors
  and cancellation leave the saved prompt intact.
- Draft edits made during preview generation are kept. The older request's
  completed result is saved in the library instead of overwriting the new draft.

Optional inputs are `idea: STRING`, `images: IMAGE` (one image or a video frame
batch), `audio: AUDIO`, and `video: VIDEO`. A connected idea overrides the Studio
brief for both the Generate button and Generate on queue.

## References

References are sent automatically. Choose a vision-capable LLM for images and
video frames, or an audio-capable LLM/server for sound. Reference support depends
on both the LLM **and** its server. No silent fallback discards an unsupported
reference; the server's error is displayed if it rejects the input.

| Reference | Preparation and transport |
|---|---|
| Uploaded images | JPEG copy, longest side up to 1024 px; sent as `image_url` data URLs. |
| Uploaded video | 1–8 evenly sampled frames, default 4. The browser decodes and prepares them; the original video is never uploaded. Frame timestamps accompany the images. |
| Uploaded audio | Browser-decodable source up to 32 MiB; selected start and duration, up to 30 seconds, prepared as mono PCM16 WAV at 24 kHz and sent as `input_audio`. |
| Connected IMAGE batch | Automatic mode includes **every image in batches of up to 8**. Larger batches are treated as video and evenly sampled, default 4 frames. Reference sampling lets you explicitly select separate images or video frames. Only selected frames are resized/converted, one at a time, to at most 1024 px. Set reference FPS for video-batch timestamps. |
| Connected VIDEO | Samples across the clip using its own duration and frame rate, default 4 frames, up to 8. Existing trims, crops and rotation are honored. File-backed video is decoded in short windows rather than materializing the whole clip. Connect AUDIO separately to include sound. |
| Connected AUDIO | First batch item, first 30 seconds; mixed to mono and resampled to 24 kHz. |

A request allows up to **8 visual references and 1 audio excerpt** across uploads
and connected inputs. These bounds limit request/context memory; they do not
limit the duration of the video or audio prompt you are writing. Video references
describe sampled visual states, not every frame or the video's soundtrack.
Browser codec support determines which uploaded files can be prepared. Native
VIDEO inputs use ComfyUI's video decoding support.

Writing a prompt **for an audio model** works with a text-only LLM. Listening to
an audio reference requires a model/server that implements OpenAI-style
`input_audio`. An endpoint being OpenAI-compatible alone does not guarantee this.

## Profiles, library and backup

Built-in starting profiles cover general image/video/audio, FLUX/Z-Image/Krea,
SDXL, MiniMax H3, Wan/LTX, and music. They are editable writing guidance, not
claims that every target uses a mandatory prompt syntax. Add a profile for a
model's particular template in **Profiles**, with instructions and default
negative text. A workflow stores its profile snapshot so it remains usable if
the workspace profile later changes or is removed. Applying a profile resets
the draft's exclusions to that profile's defaults.

**Save to library** keeps positive and negative text together with the creative
settings, profile and selected reference IDs. Search matches the title, prompt,
negative text and profile. Filter by output type or favorites. **Reuse prompt**
restores it into the current node without calling the LLM.

**Workspace → Export workspace JSON** includes profiles, connection settings,
library entries and prepared reference media. API keys are excluded. Restore
validates the complete backup before replacing the workspace. Existing keys are
retained only for matching connection IDs with unchanged endpoints. On another
machine, re-enter keys or set the named environment variables.

The workspace is shared by users/workflows on this ComfyUI installation, in
`<ComfyUI user directory>/__apple_silicon_prompt_studio/`. This is intended for
a trusted local ComfyUI installation, not as a multi-tenant credential vault.
The active draft stays in the ComfyUI workflow; save it to the library to include
it in a workspace backup. Backups contain prepared uploads, not original files.
Connected input media is sent for that request and is not copied into the Studio
library or workspace backup; its source nodes remain part of the workflow.
The workspace limit is 48 MiB. Workspace provides reference cleanup; media used
by saved library entries cannot be deleted until those entries are removed.

## Validation

Validated on ComfyUI 0.37.0, frontend 1.53.6 and the existing PyTorch 2.14 install:

- 20 backend tests: credentials, endpoint changes, backup/restore, media bounds,
  model response parsing, real local HTTP transport, cancellation, saved text,
  inclusion of all five connected images, resizing, bounded VIDEO sampling,
  persistent model discovery and discarding caches when endpoints change.
- 5 frontend execution tests: pruning to upstream inputs, target resolution,
  repeat generation, cancellation restricted to the current job and error display.
- Actual queued workflows: connected image/audio/idea inputs, two successive
  fresh generations, and exact saved-string delivery to a downstream node.
- Browser checks in classic mode and Nodes 2.0: connection discovery, editable
  image/video/audio results, sampled MP4 frames, image and WAV uploads, library
  search/favorites/reuse, and workflow execution.
- Connected Generate checks in an isolated ComfyUI: all five images sent together
  with the brief, repeated clicks, four samples from a trimmed native VIDEO,
  sampling a 20-frame IMAGE batch, text-only generation, upstream failure,
  scoped cancellation and preserving a draft edited while generation runs.
  The downstream sentinel stayed at zero during button tests; a normal Run then
  delivered the saved string to it unchanged. Only synthetic media was used.
- Usability checks in both renderers: model changes reach the outgoing request,
  the queue toggle switches between fresh generation and saved text, connection
  discovery updates the canvas picker, and new nodes retain cached choices when
  the LLM server is offline. The canvas branding header is removed.

Automated integration tests use a deterministic local OpenAI-compatible test
server. A separate live oMLX check with synthetic shapes confirmed image-grounded
output from `gemma-4-26B-A4B-it-qat-mxfp4`. The text-only
`supergemma4-26b-uncensored-mlx-4bit-v2` ignored images despite successful HTTP
responses, so choose a vision-enabled model for references. No diffusion
benchmarks or kernel settings are changed.

Provider references: [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility),
[LM Studio chat completions](https://lmstudio.ai/docs/developer/openai-compat/chat-completions),
[oMLX](https://omlx.ai/).
