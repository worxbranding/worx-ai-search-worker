// Minimal ambient type declarations to satisfy TypeScript in this project

declare interface VectorizeIndex {
  // Accept either a raw vector array or an object form (vectorId) used by the binding
  query(arg: number[] | { vectorId: string } | any, opts?: any): Promise<{ matches?: Array<any> }>;
}

declare interface Ai {
  // run returns a model response; shape varies by provider so keep `any`
  run(model: any, input: any): Promise<any>;
}

declare interface KVNamespace {
  get<T = string>(key: string, type?: "text" | "json"): Promise<T | null>;
  put?(key: string, value: string, opts?: any): Promise<void>;
  // list / other methods can be added if needed
}

// (no exports) keep these global ambient declarations
