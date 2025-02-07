// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

contract HashRegistryWithBlockID {
    struct HashInfo {
        uint256 id;
        bytes32 hashValue;
    }

    mapping(uint256 => HashInfo) public idToHash;
    mapping(bytes32 => bool) public hashExists;
    uint256 public nextId;

    constructor() {
        nextId = 1;
    }

    function storeHash(bytes32 _hash) public {
        require(!hashExists[_hash], "Hash already exists");
        idToHash[nextId] = HashInfo(nextId, _hash);
        hashExists[_hash] = true;
        nextId++;
    }

    function getHashById(uint256 _id) public view returns (uint256, bytes32) {
        require(idToHash[_id].id != 0, "ID not found");
        return (idToHash[_id].id, idToHash[_id].hashValue);
    }

    function getTotalHashes() public view returns (uint256) {
        return nextId - 1;
    }
}
