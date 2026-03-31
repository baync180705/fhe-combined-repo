import * as dotenv from "dotenv";
import { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-ethers";
import "@cofhe/hardhat-plugin";

dotenv.config();

const deployerKey = process.env.PRIVATE_KEY ?? process.env.DEPLOYER_PRIVATE_KEY;

const accounts = deployerKey
  ? [deployerKey]
  : [];

const config: HardhatUserConfig = {
  solidity: {
    version: "0.8.28",
    settings: {
      evmVersion: "cancun",
    },
  },
  cofhe: {
    logMocks: true,
    gasWarning: true,
  },
  networks: {
    "eth-sepolia": {
      url: process.env.SEPOLIA_RPC_URL ?? "",
      chainId: 11155111,
      accounts,
    },
  },
};

export default config;
