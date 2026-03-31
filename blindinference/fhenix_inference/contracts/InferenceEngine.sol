// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "@fhenixprotocol/cofhe-contracts/FHE.sol";

interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
}

interface IModelRegistry {
    function getWeightHandles(uint256 modelId) external view returns (euint32[] memory);
    function getBiasHandle(uint256 modelId) external view returns (euint32);
    function getInferenceFee(uint256 modelId) external view returns (uint256);
    function getLabWallet(uint256 modelId) external view returns (address);
}

interface IPaymentEscrow {
    function lockFeeFromInference(
        address requester,
        uint256 modelId,
        uint256 amount,
        address labWallet
    ) external returns (uint256);
    function markProcessing(uint256 requestId) external;
    function release(uint256 requestId) external;
    function escrows(uint256 requestId) external view returns (
        address requester,
        uint256 modelId,
        uint256 amount,
        address labWallet,
        uint8 status,
        uint256 timestamp
    );
}

contract InferenceEngine {
    address public owner;
    IModelRegistry public registry;
    IPaymentEscrow public escrow;
    IERC20 public paymentToken;

    uint256 public constant MAX_WEIGHTS_PER_BLOCK = 15;

    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;
    uint256 private _status;

    mapping(uint256 => euint32) private results;

    event InferenceCompleted(uint256 indexed requestId, address indexed requester);

    modifier nonReentrant() {
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");
        _status = _ENTERED;
        _;
        _status = _NOT_ENTERED;
    }

    constructor(address registryAddress, address escrowAddress, address tokenAddress) {
        owner = msg.sender;
        registry = IModelRegistry(registryAddress);
        escrow = IPaymentEscrow(escrowAddress);
        paymentToken = IERC20(tokenAddress);
        _status = _NOT_ENTERED;
    }

    function predict(uint256 modelId, InEuint32[] calldata encryptedInputs) external virtual nonReentrant returns (uint256) {
        return _predict(modelId, encryptedInputs);
    }

    function _predict(uint256 modelId, InEuint32[] calldata encryptedInputs) internal returns (uint256) {
        uint256 inferenceFee = registry.getInferenceFee(modelId);
        address labWallet = registry.getLabWallet(modelId);

        if (inferenceFee > 0) {
            require(
                paymentToken.transferFrom(msg.sender, address(escrow), inferenceFee),
                "Token transfer failed. Check allowance."
            );
        }

        uint256 requestId = escrow.lockFeeFromInference(msg.sender, modelId, inferenceFee, labWallet);
        escrow.markProcessing(requestId);

        euint32 finalScore = _computeResult(modelId, encryptedInputs);
        FHE.allowThis(finalScore);
        FHE.allow(finalScore, msg.sender);

        results[requestId] = finalScore;

        escrow.release(requestId);
        emit InferenceCompleted(requestId, msg.sender);
        return requestId;
    }

    function _computeResult(uint256 modelId, InEuint32[] calldata encryptedInputs) internal returns (euint32) {
        euint32[] memory weights = registry.getWeightHandles(modelId);
        euint32 bias = registry.getBiasHandle(modelId);

        require(weights.length > 0, "Model has no weights");
        require(weights.length <= MAX_WEIGHTS_PER_BLOCK, "Model exceeds FHE block gas limit");
        require(weights.length == encryptedInputs.length, "Input and Weight dimension mismatch");

        // Keep inference on native euint32 lanes so coFHE can accelerate the dot product and activation.
        euint32 sum = FHE.asEuint32(0);
        for (uint256 i = 0; i < weights.length; i++) {
            euint32 encryptedInput = FHE.asEuint32(encryptedInputs[i]);
            sum = FHE.add(sum, FHE.mul(encryptedInput, weights[i]));
        }

        return FHE.add(sum, bias);
    }

    function getResult(uint256 requestId) external view returns (euint32) {
        (address requester, , , , , ) = escrow.escrows(requestId);
        require(msg.sender == requester, "Not authorized to view this result");
        require(FHE.isInitialized(results[requestId]), "Result not ready or failed");

        return results[requestId];
    }
}
