// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "@fhenixprotocol/cofhe-contracts/FHE.sol";

contract ModelRegistry {
    address public owner;
    address public inferenceEngine;

    struct AILab {
        string profileURI;
        bool isRegistered;
    }

    struct Model {
        uint256 modelId;
        address labWallet;
        uint256 inferenceFee;
        string ipfsHash;
        euint32[] weightHandles;
        euint32 biasHandle;
        bool active;
    }

    mapping(address => AILab) public aiLabs;
    mapping(uint256 => Model) public models;
    uint256 public modelCount;

    event LabRegistered(address indexed labWallet, string profileURI);
    event ModelRegistered(uint256 indexed modelId, address indexed labWallet, uint256 inferenceFee);
    event ModelDeactivated(uint256 indexed modelId);
    event PriceUpdated(uint256 indexed modelId, uint256 newPrice);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not registry owner");
        _;
    }

    modifier onlyLab(uint256 modelId) {
        require(msg.sender == models[modelId].labWallet, "Not the AI lab");
        _;
    }

    modifier onlyRegisteredLab() {
        require(aiLabs[msg.sender].isRegistered, "AI lab not registered");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function setInferenceEngine(address engine) external onlyOwner {
        inferenceEngine = engine;
    }

    function registerLab(string calldata profileURI) external {
        require(bytes(profileURI).length > 0, "Profile URI required");

        aiLabs[msg.sender] = AILab({
            profileURI: profileURI,
            isRegistered: true
        });

        emit LabRegistered(msg.sender, profileURI);
    }

    function registerModel(
        InEuint32[] calldata encryptedWeights,
        InEuint32 calldata encryptedBias,
        uint256 inferenceFee,
        string calldata ipfsHash
    ) external onlyRegisteredLab returns (uint256) {
        require(inferenceEngine != address(0), "Inference engine not linked");

        modelCount++;
        uint256 newModelId = modelCount;
        Model storage newModel = models[newModelId];

        newModel.modelId = newModelId;
        newModel.labWallet = msg.sender;
        newModel.inferenceFee = inferenceFee;
        newModel.ipfsHash = ipfsHash;
        newModel.active = true;

        for (uint256 i = 0; i < encryptedWeights.length; i++) {
            euint32 weightHandle = FHE.asEuint32(encryptedWeights[i]);
            FHE.allowThis(weightHandle);
            FHE.allow(weightHandle, inferenceEngine);
            newModel.weightHandles.push(weightHandle);
        }

        euint32 bias = FHE.asEuint32(encryptedBias);
        FHE.allowThis(bias);
        FHE.allow(bias, inferenceEngine);
        newModel.biasHandle = bias;

        emit ModelRegistered(newModelId, msg.sender, inferenceFee);
        return newModelId;
    }

    function deactivateModel(uint256 modelId) external onlyLab(modelId) {
        models[modelId].active = false;
        emit ModelDeactivated(modelId);
    }

    function updatePrice(uint256 modelId, uint256 newPrice) external onlyLab(modelId) {
        models[modelId].inferenceFee = newPrice;
        emit PriceUpdated(modelId, newPrice);
    }

    function getWeightHandles(uint256 modelId) external view returns (euint32[] memory) {
        require(models[modelId].active, "Model inactive");
        return models[modelId].weightHandles;
    }

    function getBiasHandle(uint256 modelId) external view returns (euint32) {
        require(models[modelId].active, "Model inactive");
        return models[modelId].biasHandle;
    }

    function getLabWallet(uint256 modelId) external view returns (address) {
        require(models[modelId].active, "Model inactive");
        return models[modelId].labWallet;
    }

    function getPrice(uint256 modelId) external view returns (uint256) {
        require(models[modelId].active, "Model inactive");
        return models[modelId].inferenceFee;
    }

    function getInferenceFee(uint256 modelId) external view returns (uint256) {
        require(models[modelId].active, "Model inactive");
        return models[modelId].inferenceFee;
    }

    function getIpfsHash(uint256 modelId) external view returns (string memory) {
        require(models[modelId].active, "Model inactive");
        return models[modelId].ipfsHash;
    }
}
