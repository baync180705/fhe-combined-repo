declare module '@cofhe/sdk' {
  export type CofheClient = any;
  export type EncryptedUint32Input = any;
  export const Encryptable: any;
  export const FheTypes: any;
}

declare module '@cofhe/sdk/adapters' {
  export const Ethers6Adapter: any;
}

declare module '@cofhe/sdk/chains' {
  export const chains: any;
}

declare module '@cofhe/sdk/web' {
  export const createCofheClient: any;
  export const createCofheConfig: any;
}

declare module 'fhenixjs-access-control' {
  export const getPermit: any;
}

declare module 'vite-plugin-wasm' {
  const wasm: any;
  export default wasm;
}
