// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.9;

contract MyContract {
    struct HashInfo {
        uint256 id;
        string[] hashValues; // Array to store multiple pHashes
    }

    mapping(uint256 => HashInfo) public idToHash;
    uint256 public nextId = 1;

    uint256 public similarityThreshold = 10;  // Set a default threshold for similarity (Hamming distance)

    // Function to store hashes
    function storeHashes(string[] memory _hashes) public {
        // Ensure there are exactly 20 hashes
        require(_hashes.length == 20, "Must provide exactly 20 hashes");

        // Compare with existing hashes to ensure no similar hashes
        for (uint256 i = 1; i < nextId; i++) {
            (, string[] memory storedHashes) = getHashById(i); // Use `_` to discard `storedId`

            bool isSimilar = true;
            for (uint256 j = 0; j < _hashes.length; j++) {
                uint256 distance = getHammingDistance(_hashes[j], storedHashes[j]);

                // If any hash exceeds the similarity threshold, the hashes are considered similar
                if (distance > similarityThreshold) {
                    isSimilar = false;
                    break;
                }
            }

            // If a similar hash set is found, revert the transaction
            if (isSimilar) {
                revert("These hashes are too similar to an existing entry.");
            }
        }

        // If no similar hashes are found, store the new hashes
        HashInfo storage info = idToHash[nextId];
        info.id = nextId;
        info.hashValues = _hashes;

        nextId++;
    }

    // Function to retrieve stored hashes by ID
    function getHashById(uint256 _id) public view returns (uint256, string[] memory) {
        require(_id < nextId, "ID not found");
        HashInfo memory info = idToHash[_id];
        return (info.id, info.hashValues);
    }

    // Function to get the total number of stored hashes
    function getTotalHashes() public view returns (uint256) {
        return nextId - 1;
    }

    // Function to compare hashes and calculate Hamming distance
    function compareHashes(uint256 _id, string[] memory _localHashes) public view returns (uint256 similarFrames) {
        require(_id < nextId, "ID not found");
        HashInfo memory storedInfo = idToHash[_id];

        uint256 totalFrames = _localHashes.length;
        require(totalFrames == storedInfo.hashValues.length, "Hash arrays must be of equal length");

        similarFrames = 0;

        // Loop through each hash and calculate the Hamming distance
        for (uint256 i = 0; i < totalFrames; i++) {
            uint256 distance = getHammingDistance(_localHashes[i], storedInfo.hashValues[i]);

            // Check if the distance is below threshold for similarity
            if (distance <= similarityThreshold) {  // Set your threshold here, for example 10
                similarFrames++;
            }
        }

        return similarFrames;
    }

    // Helper function to calculate Hamming distance between two hashes
    function getHammingDistance(string memory hash1, string memory hash2) public pure returns (uint256 distance) {
        require(bytes(hash1).length == bytes(hash2).length, "Hashes must have the same length");
        distance = 0;

        for (uint256 i = 0; i < bytes(hash1).length; i++) {
            if (bytes(hash1)[i] != bytes(hash2)[i]) {
                distance++;
            }
        }
    }
}
