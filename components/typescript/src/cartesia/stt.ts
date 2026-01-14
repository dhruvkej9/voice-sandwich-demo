import WebSocket from "ws";
import { readFileSync } from "fs";
import { writableIterator } from "../utils";
import type { VoiceAgentEvent } from "../types";

// Load SSL certificate for corporate proxy (Zscaler)
const CA_CERT_PATH = process.env.NODE_EXTRA_CA_CERTS || "/home/dhruvkejri1/certs/combined.pem";
let caCert: Buffer | undefined;
try {
  caCert = readFileSync(CA_CERT_PATH);
} catch (e) {
  console.warn("Could not load CA certificate:", CA_CERT_PATH);
}

interface CartesiaSTTOptions {
  apiKey?: string;
  model?: string;
  language?: string;
  encoding?: string;
  sampleRate?: number;
}

interface CartesiaTranscriptMessage {
  type: "transcript";
  text: string;
  is_final: boolean;
  request_id?: string;
  duration?: number;
  language?: string;
}

interface CartesiaErrorMessage {
  type: "error";
  error: string;
  request_id?: string;
}

type CartesiaMessage = CartesiaTranscriptMessage | CartesiaErrorMessage;

export class CartesiaSTT {
  apiKey: string;
  model: string;
  language: string;
  encoding: string;
  sampleRate: number;

  protected _bufferIterator = writableIterator<VoiceAgentEvent.STTEvent>();
  protected _connectionPromise: Promise<WebSocket> | null = null;

  protected get _connection(): Promise<WebSocket> {
    if (this._connectionPromise) {
      return this._connectionPromise;
    }

    this._connectionPromise = new Promise((resolve, reject) => {
      const params = new URLSearchParams({
        api_key: this.apiKey,
        model: this.model,
        language: this.language,
        encoding: this.encoding,
        sample_rate: this.sampleRate.toString(),
      });

      const url = `wss://api.cartesia.ai/stt/websocket?${params.toString()}`;
      const ws = new WebSocket(url, {
        headers: {
          "Cartesia-Version": "2024-11-13",
        },
        ca: caCert,
        rejectUnauthorized: true,
      });

      ws.on("open", () => {
        console.log("Cartesia STT WebSocket connected");
        resolve(ws);
      });

      ws.on("message", (data: WebSocket.RawData) => {
        try {
          const message: CartesiaMessage = JSON.parse(data.toString());
          
          if (message.type === "transcript") {
            if (message.is_final) {
              this._bufferIterator.push({
                type: "stt_output",
                transcript: message.text,
                ts: Date.now(),
              });
            } else {
              this._bufferIterator.push({
                type: "stt_chunk",
                transcript: message.text,
                ts: Date.now(),
              });
            }
          } else if (message.type === "error") {
            console.error("Cartesia STT error:", message.error);
            throw new Error(message.error);
          }
        } catch (error) {
          console.error("Error parsing Cartesia message:", error);
        }
      });

      ws.on("error", (error) => {
        console.error("Cartesia STT WebSocket error:", error);
        this._bufferIterator.cancel();
        reject(error);
      });

      ws.on("close", () => {
        console.log("Cartesia STT WebSocket closed");
        this._connectionPromise = null;
      });
    });

    return this._connectionPromise;
  }

  constructor(options: CartesiaSTTOptions) {
    this.apiKey = options.apiKey || process.env.CARTESIA_API_KEY || "";
    this.model = options.model || "ink-whisper";
    this.language = options.language || "en";
    this.encoding = options.encoding || "pcm_s16le";
    this.sampleRate = options.sampleRate || 16000;

    if (!this.apiKey) {
      throw new Error("Cartesia API key is required");
    }
  }

  async sendAudio(buffer: Uint8Array): Promise<void> {
    const conn = await this._connection;
    conn.send(buffer);
  }

  async *receiveEvents(): AsyncGenerator<VoiceAgentEvent.STTEvent> {
    yield* this._bufferIterator;
  }

  async close(): Promise<void> {
    if (this._connectionPromise) {
      const ws = await this._connectionPromise;
      const doneMessage = JSON.stringify({ type: "done" });
      ws.send(doneMessage);
      ws.close();
    }
  }
}
