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
        clippy-hook = pkgs.writeScript "clippy-hook" ''
          #!${pkgs.stdenv.shell}
          export CARGO_HOME="${rustToolchain}/.cargo"
          export RUSTUP_HOME="${rustToolchain}/.rustup"
          export PATH="${rustToolchain}/bin:$PATH"
          mkdir -p target
          exec ${rustToolchain}/bin/cargo clippy --all-targets --all-features -- -D warnings
        '';

        pre-commit-check = pre-commit-hooks.lib.${system}.run {
          src = ./.;
          hooks = {
            nixpkgs-fmt.enable = true;
            rustfmt.enable = true;
            ruff.enable = true;
            clippy = {
              enable = true;
              entry = toString clippy-hook;
            };
          };
          tools = {
            ruff = pkgs.ruff;
            rustfmt = rustToolchain;
            clippy = rustToolchain;
          };
        };
      in
      {
        packages = {
          default = trusty;
          pyApp = poetryApplication;
        };

        checks = {
          # primary issue was that `nix flake check` runs in a pure environment,
          # preventing Clippy and Cargo from accessing the internet or untracked files.
          # this caused failures when trying to fetch Git dependencies like `gbdt`.
          # more info: https://github.com/cachix/git-hooks.nix/issues/452
          # we replace direct pre-commit-hook with a custom mkDerivation

          pre-commit-check = pkgs.stdenv.mkDerivation {
            name = "pre-commit-check";
            src = ./.;

            nativeBuildInputs = [
              pkgs.rustPlatform.cargoSetupHook
              rustToolchain
            ];

            buildInputs = with pkgs; [
              git
              openssl
              pkg-config
              rustToolchain
            ];

            cargoDeps = pkgs.rustPlatform.importCargoLock {
              lockFile = ./Cargo.lock;
              outputHashes = {
                "gbdt-0.1.3" = "sha256-f2uqulFSNGwrDM7RPdGIW11VpJRYexektXjHxTJHHmA=";
              };
            };

            buildPhase = ''
              export CARGO_HOME="${rustToolchain}/.cargo"
              export RUSTUP_HOME="${rustToolchain}/.rustup"
              export PATH="${rustToolchain}/bin:$PATH"
              ${pre-commit-check.buildCommand}
            '';

            installPhase = ''
              touch $out
            '';

            RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
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
          shellHook = ''
            ${pre-commit-check.shellHook}
          '';
        };
      });
}
