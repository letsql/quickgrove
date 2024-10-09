{
  description = "A devShell example with Crane for Cargo builds";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, crane, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        rustToolchain = pkgs.rust-bin.beta.latest.default;
        craneLib = crane.mkLib pkgs;

        # Common derivation arguments used for all builds
        commonArgs = {
          src = craneLib.cleanCargoSource ./.;
          strictDeps = true;

          buildInputs = with pkgs; [
            openssl
            libiconv
          ];

          nativeBuildInputs = with pkgs; [
            pkg-config
          ];
        };

        # Build *just* the cargo dependencies, so we can reuse
        # all of that work (e.g. via cachix) when running in CI
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        # Build the actual crate itself, reusing the dependency
        # artifacts from above.
        trusty = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
        });

        # Run clippy (and deny all warnings) on the crate source
        trustyClippy = craneLib.cargoClippy (commonArgs // {
          inherit cargoArtifacts;
          cargoClippyExtraArgs = "--all-targets -- --deny warnings";
        });

        # Run the crate's tests
        trustyTests = craneLib.cargoTest (commonArgs // {
          inherit cargoArtifacts;
        });
      in
      {
        packages.default = trusty;

        checks = {
          inherit
            trusty
            trustyClippy
            trustyTests;
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ trusty ];
          buildInputs = with pkgs; [
            eza
            fd
            rustToolchain
            libiconv
          ];

          shellHook = ''
            alias ls=eza
            alias find=fd
          '';
        };
      });
}
