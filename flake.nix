{
  description = "A devShell with Crane for Cargo builds for trusty";

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

        rustToolchain = pkgs.rust-bin.stable.latest.default;
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
        
        customFilter = path: type:
          let baseName = baseNameOf (toString path);
          in
          (craneLib.filterCargoSources path type) ||
          (baseName == "diamonds.csv") ||
          (baseName == "pricing-model-100-mod.json" && (builtins.match ".*models.*" path) != null);

        commonArgs = {
          src = pkgs.lib.cleanSourceWith {
            src = ./.;
            filter = customFilter;
          };
          strictDeps = true;
        
          buildInputs = with pkgs; [
            openssl
          ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.libiconv
            pkgs.darwin.apple_sdk.frameworks.Security
          ];
        
          nativeBuildInputs = with pkgs; [
            pkg-config
          ];
        };

        # Build *just* the cargo dependencies
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        trusty = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
        });

        trustyClippy = craneLib.cargoClippy (commonArgs // {
          inherit cargoArtifacts;
          cargoClippyExtraArgs = "--all-targets -- --deny warnings";
        });

        trustyTests = craneLib.cargoTest (commonArgs // {
          inherit cargoArtifacts;
        });

      in
      {
        packages = {
          default = trusty;
        };

        checks = {
          inherit
            trusty
            trustyClippy
            trustyTests;
        };

        devShells.default = craneLib.devShell {
          checks = self.checks.${system};
          inputsFrom = [ trusty ];
        };
      });
}
