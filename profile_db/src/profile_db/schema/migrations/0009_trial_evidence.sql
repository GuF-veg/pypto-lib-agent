-- Candidate evidence is trial memory, independent of disposable profile runs.
CREATE TABLE trial_evidence (
    trial_id BIGINT PRIMARY KEY,
    manifest_locator VARCHAR NOT NULL,
    manifest_sha256 VARCHAR NOT NULL,
    document JSON NOT NULL
);
