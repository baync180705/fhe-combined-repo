import * as dotenv from "dotenv";
import { ethers } from "hardhat";

dotenv.config();

async function main() {
  const [deployer] = await ethers.getSigners();
  const feeTreasury = process.env.FEE_TREASURY ?? deployer.address;
  const initialSupply = ethers.parseUnits("1000000", 18);

  const tokenFactory = await ethers.getContractFactory("MockERC20", deployer);
  const paymentToken = await tokenFactory.deploy("Blindference Token", "BFHE", initialSupply);
  await paymentToken.waitForDeployment();

  const registryFactory = await ethers.getContractFactory("ModelRegistry", deployer);
  const registry = await registryFactory.deploy();
  await registry.waitForDeployment();

  const escrowFactory = await ethers.getContractFactory("PaymentEscrow", deployer);
  const escrow = await escrowFactory.deploy(
    await paymentToken.getAddress(),
    await registry.getAddress(),
    feeTreasury,
  );
  await escrow.waitForDeployment();

  const inferenceFactory = await ethers.getContractFactory("BlindInference", deployer);
  const blindInference = await inferenceFactory.deploy(
    await registry.getAddress(),
    await escrow.getAddress(),
    await paymentToken.getAddress(),
  );
  await blindInference.waitForDeployment();

  await (await registry.setInferenceEngine(await blindInference.getAddress())).wait();
  await (await escrow.setInferenceEngine(await blindInference.getAddress())).wait();

  console.log("BlindInference suite deployed");
  console.log(`Deployer: ${deployer.address}`);
  console.log(`Payment token: ${await paymentToken.getAddress()}`);
  console.log(`ModelRegistry: ${await registry.getAddress()}`);
  console.log(`PaymentEscrow: ${await escrow.getAddress()}`);
  console.log(`BlindInference: ${await blindInference.getAddress()}`);
  console.log(`Fee treasury: ${feeTreasury}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
