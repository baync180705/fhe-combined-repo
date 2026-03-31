import * as dotenv from "dotenv";
import { ethers } from "hardhat";
import hre from "hardhat";
import { Encryptable, FheTypes } from "@cofhe/sdk";

dotenv.config();

async function main() {
  const contractAddress = process.env.BLIND_INFERENCE_ADDRESS;
  if (!contractAddress) {
    throw new Error("BLIND_INFERENCE_ADDRESS is not set");
  }
  const requestId = BigInt(process.env.REQUEST_ID ?? "1");

  const [user] = await ethers.getSigners();
  const contract = await ethers.getContractAt("BlindInference", contractAddress, user);
  const cofheClient = await hre.cofhe.createClientWithBatteries(user);

  const scaledGlucose = 115;
  const scaledBmi = 133;

  const encryptedFeatures = await cofheClient
    .encryptInputs([
      Encryptable.uint32(BigInt(scaledGlucose)),
      Encryptable.uint32(BigInt(scaledBmi)),
    ])
    .execute();

  const tx = await contract.predict(requestId, encryptedFeatures);
  const receipt = await tx.wait();

  const encryptedPrediction = await contract.getResult(requestId);
  const unsealedPrediction = await cofheClient
    .decryptForView(encryptedPrediction, FheTypes.Uint32)
    .execute();

  console.log("Prediction transaction mined");
  console.log(`User: ${user.address}`);
  console.log(`Tx hash: ${receipt?.hash ?? "unknown"}`);
  console.log(`Request ID: ${requestId}`);
  console.log(`Scaled inputs: glucose=${scaledGlucose}, bmi=${scaledBmi}`);
  console.log(`Predicted positive class: ${unsealedPrediction === 1n}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
