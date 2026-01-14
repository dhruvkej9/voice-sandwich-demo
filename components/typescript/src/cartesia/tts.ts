import WebSocket from "ws";
import { readFileSync } from "fs";
import { writableIterator } from "../utils";
import type {
  CartesiaTTSRequest,
  CartesiaTTSResponse,
  CartesiaOutputFormat,
  CartesiaVoice,
} from "./api-types";
import type { VoiceAgentEvent } from "../types";

// Load SSL certificate for corporate proxy (Zscaler)
const CA_CERT_PATH = process.env.NODE_EXTRA_CA_CERTS || "/home/dhruvkejri1/certs/combined.pem";
let caCert: Buffer | undefined;
try {
  caCert = readFileSync(CA_CERT_PATH);
} catch (e) {
  console.warn("Could not load CA certificate:", CA_CERT_PATH);
}

interface CartesiaTTSOptions {
  apiKey?: string;
  voiceId?: string;
  modelId?: string;
  sampleRate?: number;
  encoding?: CartesiaOutputFormat["encoding"];
  language?: string;
  cartesiaVersion?: string;
}

export class CartesiaTTS {
  apiKey: string;
  voiceId: string;
  modelId: string;
  sampleRate: number;
  encoding: CartesiaOutputFormat["encoding"];
  language: string;
  cartesiaVersion: string;

  protected _bufferIterator = writableIterator<VoiceAgentEvent.TTSChunk>();
  protected _ws: WebSocket | null = null;
  protected _connectionPromise: Promise<WebSocket> | null = null;
  protected _contextCounter = 0;
  protected _isClosing = false;
  protected _pendingRequests = 0;

  /**
   * Generate a valid context_id for Cartesia.
   * Context IDs must only contain alphanumeric characters, underscores, and hyphens.
   */
  protected _generateContextId(): string {
    const timestamp = Date.now();
    const counter = this._contextCounter++;
    return `ctx_${timestamp}_${counter}`;
  }

  protected _connect(): Promise<WebSocket> {
    // Return existing connection if available and open
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      return Promise.resolve(this._ws);
    }

    // Return pending connection promise if connecting
    if (this._connectionPromise) {
      return this._connectionPromise;
    }

    this._connectionPromise = new Promise((resolve, reject) => {
      const params = new URLSearchParams({
        api_key: this.apiKey,
        cartesia_version: this.cartesiaVersion,
      });
      const url = `wss://api.cartesia.ai/tts/websocket?${params.toString()}`;
      
      console.log("Cartesia TTS: Connecting...");
      const ws = new WebSocket(url, {
        ca: caCert,
        rejectUnauthorized: true,
      });

      ws.on("open", () => {
        console.log("Cartesia TTS: WebSocket connected");
        this._ws = ws;
        this._connectionPromise = null;
        resolve(ws);
      });

      ws.on("message", (data: WebSocket.RawData) => {
        try {
          const message: CartesiaTTSResponse = JSON.parse(data.toString());

          if (message.data) {
            this._bufferIterator.push({
              type: "tts_chunk",
              audio: message.data,
              ts: Date.now(),
            });
          } else if (message.done) {
            // Audio generation complete for this context
            this._pendingRequests = Math.max(0, this._pendingRequests - 1);
            console.log(`Cartesia TTS: Context complete, pending=${this._pendingRequests}`);
          } else if (message.error) {
            console.error(`Cartesia TTS error: ${message.error}`);
            // Don't throw - log and continue, let close() handle cleanup
          }
        } catch (error) {
          console.error("Cartesia TTS: Failed to parse message:", error);
        }
      });

      ws.on("error", (error) => {
        console.error("Cartesia TTS: WebSocket error:", error.message);
        this._connectionPromise = null;
        this._ws = null;
        reject(error);
      });

      ws.on("close", (code, reason) => {
        console.log(`Cartesia TTS: WebSocket closed (code=${code}, reason=${reason || "none"})`);
        this._connectionPromise = null;
        this._ws = null;
        // Only cancel iterator if we're intentionally closing
        if (this._isClosing) {
          this._bufferIterator.cancel();
        }
      });
    });

    return this._connectionPromise;
  }

  constructor(options: CartesiaTTSOptions = {}) {
    this.apiKey = options.apiKey ?? process.env.CARTESIA_API_KEY ?? "";
    if (!this.apiKey) {
      throw new Error("Cartesia API key is required");
    }
    this.voiceId = options.voiceId ?? "f6ff7c0c-e396-40a9-a70b-f7607edb6937";
    this.modelId = "sonic-3";
    this.sampleRate = 24000;
    this.encoding = "pcm_s16le";
    this.language = "en";
    this.cartesiaVersion = options.cartesiaVersion ?? "2025-04-16";
  }

  async sendText(text: string): Promise<void> {
    if (!text || !text.trim()) {
      return;
    }

    if (this._isClosing) {
      console.warn("Cartesia TTS: Cannot send text, connection is closing");
      return;
    }

    try {
      const conn = await this._connect();
      
      if (conn.readyState !== WebSocket.OPEN) {
        console.warn("Cartesia TTS: Connection not open, cannot send text");
        return;
      }

      const voice: CartesiaVoice = {
        mode: "id",
        id: this.voiceId,
      };

      const outputFormat: CartesiaOutputFormat = {
        container: "raw",
        encoding: this.encoding,
        sample_rate: this.sampleRate,
      };

      const contextId = this._generateContextId();
      const payload: CartesiaTTSRequest = {
        model_id: this.modelId,
        transcript: text,
        voice: voice,
        output_format: outputFormat,
        language: this.language,
        context_id: contextId,
      };

      this._pendingRequests++;
      console.log(`Cartesia TTS: Sending text (context=${contextId}, pending=${this._pendingRequests})`);
      conn.send(JSON.stringify(payload));
    } catch (error) {
      console.error("Cartesia TTS: Failed to send text:", error);
      throw error;
    }
  }

  async *receiveEvents(): AsyncGenerator<VoiceAgentEvent.TTSChunk> {
    yield* this._bufferIterator;
  }

  async close(): Promise<void> {
    console.log("Cartesia TTS: Closing connection...");
    this._isClosing = true;
    
    // Wait a bit for pending audio to arrive
    if (this._pendingRequests > 0) {
      console.log(`Cartesia TTS: Waiting for ${this._pendingRequests} pending requests...`);
      await new Promise(resolve => setTimeout(resolve, 500));
    }

    // Close the WebSocket
    if (this._ws) {
      if (this._ws.readyState === WebSocket.OPEN) {
        this._ws.close(1000, "Normal closure");
      }
      this._ws = null;
    }

    this._connectionPromise = null;
    this._bufferIterator.cancel();
    console.log("Cartesia TTS: Connection closed");
  }
}
