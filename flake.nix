{
  description = "A devShell for poetry and cargo for trusty";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix.url = "github:nix-community/poetry2nix";
    pre-commit-hooks = {
      url = "github:cachix/pre-commit-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = { self, nixpkgs, rust-overlay, crane, flake-utils, poetry2nix, pre-commit-hooks, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [
          (import rust-overlay)
          poetry2nix.overlays.default
        ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default;
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
        src = craneLib.cleanCargoSource ./.;

        allowedExtensions = [
          "csv"
          "json"
        ];

        hasAllowedExtension = path:
          let
            extension = pkgs.lib.lists.last (pkgs.lib.strings.splitString "." (baseNameOf (toString path)));
          in
          pkgs.lib.lists.any (ext: ext == extension) allowedExtensions;

        customFilter = path: type:
          let
            isCargoSource = craneLib.filterCargoSources path type;
            isAllowed = type == "regular" && hasAllowedExtension path;
          in
          isCargoSource || isAllowed;

        commonArgs = {
          src = pkgs.lib.cleanSourceWith {
            src = ./.;
            filter = customFilter;
          };
          strictDeps = true;
          CARGO_NET_OFFLINE = "false";
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

        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        trusty = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
        });

        poetryApplication = pkgs.poetry2nix.mkPoetryApplication {
          projectDir = ./.;
          preferWheels = true;
          overrides = pkgs.poetry2nix.overrides.withDefaults
            (self: super: {
              atpublic = super.atpublic.overridePythonAttrs
                (
                  old: {
                    buildInputs = (old.buildInputs or [ ]) ++ [ super.hatchling ];
                  }
                );
              xgboost = super.xgboost.overridePythonAttrs (old: { } // pkgs.lib.attrsets.optionalAttrs pkgs.stdenv.isDarwin {
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ super.cmake ];
                cmakeDir = "../cpp_src";
                preBuild = ''
                  cd ..
                '';
              });
            });
        };
        pythonEnv = poetryApplication.dependencyEnv;
      in
      {
        packages = {
          default = trusty;
          pyApp = poetryApplication;
        };
        checks = {
          pre-commit-check = pre-commit-hooks.lib.${system}.run {
            src = ./.;
            hooks = {
              nixpkgs-fmt.enable = true;
              rustfmt.enable = true;
              ruff.enable = true;
              # todo: It seems like that everyting is built offline in the checks
              # and clippy cannot pull data from gbdt-rs git repo. 
              # error: failed to get `gbdt` as a dependency of package `trusty v0.1.0 (/private/tmp/nix-build-pre-commit-run.drv-0/src)`
              # clippy.enable = true;
            };
            tools = { ruff = pkgs.ruff; };
          };
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ trusty ];
          buildInputs = [
            rustToolchain
            pythonEnv
            pkgs.poetry
            # add pre-commit dependencies
            pkgs.ruff
            pkgs.rustfmt
            pkgs.nixpkgs-fmt
          ];
          # automatically set up git hooks
          inherit (self.checks.${system}.pre-commit-check) shellHook;
        };
      });
}
