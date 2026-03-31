// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "./InferenceEngine.sol";

contract BlindInference is InferenceEngine {
    event PredictionRequested(uint256 indexed requestId, address indexed requester, uint256 indexed modelId);

    constructor(address registryAddress, address escrowAddress, address tokenAddress)
        InferenceEngine(registryAddress, escrowAddress, tokenAddress)
    {}

    function predict(uint256 modelId, InEuint32[] calldata encryptedInputs) external override returns (uint256) {
        uint256 requestId = _predict(modelId, encryptedInputs);
        emit PredictionRequested(requestId, msg.sender, modelId);
        return requestId;
    }
}
