// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function transfer(address recipient, uint256 amount) external returns (bool);
}

interface IModelRegistryForEscrow {
    function getPrice(uint256 modelId) external view returns (uint256);
    function getLabWallet(uint256 modelId) external view returns (address);
}

contract PaymentEscrow {
    address public owner;
    address public inferenceEngine;
    address public feeTreasury;

    IERC20 public paymentToken;
    IModelRegistryForEscrow public registry;

    uint256 public protocolFeeBps = 250;
    uint256 public constant TIMEOUT_DURATION = 24 hours;
    uint256 public requestCount;

    enum RequestStatus {
        NONE,
        PENDING,
        PROCESSING,
        COMPLETED,
        REFUNDED
    }

    struct EscrowRecord {
        address requester;
        uint256 modelId;
        uint256 amount;
        address labWallet;
        RequestStatus status;
        uint256 timestamp;
    }

    mapping(uint256 => EscrowRecord) public escrows;

    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;
    uint256 private _status;

    modifier nonReentrant() {
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");
        _status = _ENTERED;
        _;
        _status = _NOT_ENTERED;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyEngine() {
        require(msg.sender == inferenceEngine, "Only InferenceEngine");
        _;
    }

    event FeeLocked(uint256 indexed requestId, address indexed requester, uint256 indexed modelId, uint256 amount);
    event ProcessingStarted(uint256 indexed requestId);
    event FeeReleased(uint256 indexed requestId, address indexed labWallet, uint256 labAmount, uint256 protocolAmount);
    event FeeRefunded(uint256 indexed requestId, address indexed requester, uint256 amount);

    constructor(address paymentTokenAddress, address registryAddress, address treasury) {
        owner = msg.sender;
        feeTreasury = treasury;
        paymentToken = IERC20(paymentTokenAddress);
        registry = IModelRegistryForEscrow(registryAddress);
        _status = _NOT_ENTERED;
    }

    function setInferenceEngine(address engine) external onlyOwner {
        inferenceEngine = engine;
    }

    function setProtocolFee(uint256 bps) external onlyOwner {
        require(bps <= 1000, "Fee cannot exceed 10%");
        protocolFeeBps = bps;
    }

    function lockFee(uint256 modelId) external nonReentrant returns (uint256) {
        uint256 price = registry.getPrice(modelId);
        address lab = registry.getLabWallet(modelId);

        requestCount++;
        uint256 requestId = requestCount;

        escrows[requestId] = EscrowRecord({
            requester: msg.sender,
            modelId: modelId,
            amount: price,
            labWallet: lab,
            status: RequestStatus.PENDING,
            timestamp: block.timestamp
        });

        if (price > 0) {
            require(
                paymentToken.transferFrom(msg.sender, address(this), price),
                "Token transfer failed. Check allowance."
            );
        }

        emit FeeLocked(requestId, msg.sender, modelId, price);
        return requestId;
    }

    function lockFeeFromInference(
        address requester,
        uint256 modelId,
        uint256 amount,
        address labWallet
    ) external onlyEngine nonReentrant returns (uint256) {
        requestCount++;
        uint256 requestId = requestCount;

        escrows[requestId] = EscrowRecord({
            requester: requester,
            modelId: modelId,
            amount: amount,
            labWallet: labWallet,
            status: RequestStatus.PENDING,
            timestamp: block.timestamp
        });

        emit FeeLocked(requestId, requester, modelId, amount);
        return requestId;
    }

    function markProcessing(uint256 requestId) external onlyEngine {
        require(escrows[requestId].status == RequestStatus.PENDING, "Request not PENDING");
        escrows[requestId].status = RequestStatus.PROCESSING;
        emit ProcessingStarted(requestId);
    }

    function release(uint256 requestId) external onlyEngine nonReentrant {
        EscrowRecord storage record = escrows[requestId];
        require(record.status == RequestStatus.PROCESSING, "Request not PROCESSING");

        record.status = RequestStatus.COMPLETED;

        uint256 totalAmount = record.amount;
        if (totalAmount > 0) {
            uint256 protocolCut = (totalAmount * protocolFeeBps) / 10000;
            uint256 labCut = totalAmount - protocolCut;

            if (protocolCut > 0) {
                require(paymentToken.transfer(feeTreasury, protocolCut), "Protocol fee transfer failed");
            }
            if (labCut > 0) {
                require(paymentToken.transfer(record.labWallet, labCut), "Lab fee transfer failed");
            }

            emit FeeReleased(requestId, record.labWallet, labCut, protocolCut);
        } else {
            emit FeeReleased(requestId, record.labWallet, 0, 0);
        }
    }

    function refund(uint256 requestId) external nonReentrant {
        EscrowRecord storage record = escrows[requestId];
        require(record.requester == msg.sender, "Not your request");
        require(record.status == RequestStatus.PENDING, "Cannot refund: Processing started");

        record.status = RequestStatus.REFUNDED;

        if (record.amount > 0) {
            require(paymentToken.transfer(msg.sender, record.amount), "Refund transfer failed");
        }

        emit FeeRefunded(requestId, msg.sender, record.amount);
    }

    function forceRefund(uint256 requestId) external nonReentrant {
        EscrowRecord storage record = escrows[requestId];
        require(record.requester == msg.sender, "Not your request");
        require(record.status == RequestStatus.PROCESSING, "Not PROCESSING");
        require(block.timestamp > record.timestamp + TIMEOUT_DURATION, "Timeout period not reached");

        record.status = RequestStatus.REFUNDED;

        if (record.amount > 0) {
            require(paymentToken.transfer(msg.sender, record.amount), "Force refund failed");
        }

        emit FeeRefunded(requestId, msg.sender, record.amount);
    }

    function getRequestStatus(uint256 requestId) external view returns (RequestStatus) {
        return escrows[requestId].status;
    }
}
