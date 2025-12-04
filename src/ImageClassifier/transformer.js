import { pipeline } from "@huggingface/transformers";
import handleArguments, { isVideo } from "../utils/handleArguments";
import { drawToCanvas } from "../utils/imageUtilities";

function chooiseDevice() {
  if (typeof navigator !== "undefined" && navigator.gpu) return "webgpu";
  return "wasm";
}

/**
 * @reference https://huggingface.co/docs/transformers.js/en/api/pipelines#module_pipelines.ImageClassificationPipeline
 */
export class ImageClassifierTransformer {
  constructor(options, callback) {
    this.classifier = null;
    this.needToStop = false;
    this.isClassifying = false;
    this.topK = options.topK || 3;
    this.device = chooiseDevice();

    this.ready = pipeline(
      "image-classification",
      "Xenova/vit-base-patch16-224",
      { device: this.device }
    ).then((classifier) => {
      this.classifier = classifier;
      callback?.(this);
      return this;
    });
  }

  async classify(inputNumOrCallback, numOrCallback, cb) {
    if (this.isClassifying || !this.classifier) return;
    this.isClassifying = true;
    const { image, number, callback } = handleArguments(
      inputNumOrCallback,
      numOrCallback,
      cb
    ).require(
      "image",
      "No input image provided. If you want to classify a video, use classifyStart."
    );

    // Transfomerjs doesn't support HTMLVideoElement directly, so convert to canvas
    const input = isVideo(image) ? drawToCanvas(image) : image;

    // Convert topK to top_k for transfomerjs and get the results
    const topK = number !== undefined ? number : this.topK;
    const results = await this.classifier(input, { top_k: topK });

    // Normalize the results to match the format form tensorflowjs
    const normalized = results.map((result) => ({
      label: result.label,
      confidence: result.score,
    }));

    callback(normalized);
    this.isClassifying = false;
    return normalized;
  }

  async classifyStart(inputNumOrCallback, numOrCallback, cb) {
    if (this.isClassifying || !this.classifier) return;

    this.needToStop = false;

    const next = () => {
      if (this.needToStop) return;
      this.classify(inputNumOrCallback, numOrCallback, cb).then(() => {
        // WebGPU is very fast, so we can call the next frame immediately
        if (this.device === "webgpu") next();
        // Wasm is very slow, so we need to wait for 1 second before calling the next frame
        else setTimeout(next, 1000);
      });
    };

    next();
  }

  async classifyStop() {
    this.needToStop = true;
  }
}
