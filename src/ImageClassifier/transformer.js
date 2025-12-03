import { pipeline } from "@huggingface/transformers";
import handleArguments from "../utils/handleArguments";

/**
 * @reference https://huggingface.co/docs/transformers.js/en/api/pipelines#module_pipelines.ImageClassificationPipeline
 */
export class ImageClassifierTransformer {
  constructor(options, callback) {
    this.classifier = null;
    this.needToStop = false;
    this.isClassifying = false;
    this.ready = pipeline(
      "image-classification",
      "Xenova/vit-base-patch16-224",
      options
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
    const options = number !== undefined ? { top_k: number } : {};
    const results = await this.classifier(image, options);
    // Normalize the results to match the format form tensorflowjs
    const normalized = results.map((result) => ({
      label: result.label,
      confidence: result.score,
    }));
    callback(normalized);
    this.isClassifying = false;
    return normalized;
  }

  async classifyStart(input, callback) {
    if (this.isClassifying || !this.classifier) return;
    this.needToStop = false;
    const next = (...args) => {
      if (this.needToStop) return;
      callback(...args);
      this.classify(input, next);
    };
    next();
  }

  async classifyStop() {
    this.needToStop = true;
  }
}
