{
  description = "A devShell for uv and cargo for trusty";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
    pre-commit-hooks = {
      url = "github:cachix/pre-commit-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        nixpkgs.follows = "nixpkgs";
      };
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        uv2nix.follows = "uv2nix";
        nixpkgs.follows = "nixpkgs";
      };
    };
  };
  outputs = { self, nixpkgs, rust-overlay, crane, flake-utils, pre-commit-hooks, pyproject-nix, uv2nix, pyproject-build-systems, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [
          (import rust-overlay)
        ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default;
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        src =
          let
            inherit (pkgs.lib.path) append;
            inherit (pkgs.lib.fileset) unions toSource fileFilter;
            inherit (pkgs.lib.lists) any;
            allowedExtensions = [ "rs" "toml" "csv" "json" ];
            hasAllowedExtension = file: any (ext: file.hasExt ext) allowedExtensions;
            _src = ./.;
          in
          toSource (unions [
            (append _src "Cargo.toml")
            (append _src "Cargo.lock")
            (fileFilter hasAllowedExtension _src)
          ]);
        cargoDeps = pkgs.rustPlatform.importCargoLock {
          lockFile = ./Cargo.lock;
          outputHashes = {
            "gbdt-0.1.3" = "sha256-f2uqulFSNGwrDM7RPdGIW11VpJRYexektXjHxTJHHmA=";
          };
        };
        commonArgs = {
          inherit cargoDeps;
          inherit src;
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

        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };
        editableOverlay = _final: prev: {
          trusty = prev.trusty.override (_: {
            # we can't use mkEditablePyprojectOverlay: it tries to append "src/"
            editableRoot = "$REPO_ROOT/python";
          });
        };
        maybeMacosOverrides = final: prev: pkgs.lib.optionalAttrs pkgs.stdenv.isDarwin {
          scikit-learn = prev.scikit-learn.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ final.resolveBuildSystem {
              meson-python = [ ];
              ninja = [ ];
              cython = [ ];
              numpy = [ ];
              scipy = [ ];
            };
          });
          scipy = prev.scipy.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ (final.resolveBuildSystem {
              meson-python = [ ];
              ninja = [ ];
              cython = [ ];
              pythran = [ ];
              pybind11 = [ ];
            }) ++ [
              pkgs.gfortran
              pkgs.cmake
              pkgs.xsimd
              pkgs.pkg-config
              pkgs.xcbuild.xcrun
              pkgs.openblas
            ];
          });
          xgboost = prev.xgboost.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ (
              final.resolveBuildSystem {
                hatchling = [ ];
              }) ++ [
              pkgs.cmake
            ];
          });
        };
        pyprojectOverrides = _final: prev: {
          trusty = prev.trusty.overrideAttrs (old: {
            inherit cargoDeps;
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
              pkgs.rustPlatform.cargoSetupHook
              pkgs.rustPlatform.maturinBuildHook
              rustToolchain
            ];
          });
        };
        pyprojectOverrides-editable = _final: prev: {
          trusty = prev.trusty.overrideAttrs (old: {
            inherit cargoDeps;
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
              rustToolchain
            ];
          });
        };
        python = pkgs.python312;
        pythonSet =
          # Use base package set from pyproject.nix builders
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          }).overrideScope
            (
              pkgs.lib.composeManyExtensions [
                pyproject-build-systems.overlays.default
                overlay
                maybeMacosOverrides
                pyprojectOverrides
              ]
            );
        pythonSet-editable =
          # Use base package set from pyproject.nix builders
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          }).overrideScope
            (
              pkgs.lib.composeManyExtensions [
                pyproject-build-systems.overlays.default
                overlay
                maybeMacosOverrides
                pyprojectOverrides-editable
                editableOverlay
              ]
            );
        venv-312 = pythonSet.mkVirtualEnv "trusty-venv" workspace.deps.all;
        venv-editable-312 = pythonSet-editable.mkVirtualEnv "trusty-editable-venv" workspace.deps.all;
        maybeMaturinBuildHook = ''
          set -eu

          repo_dir=$(git rev-parse --show-toplevel)
          name=trusty
          if [ "$(basename "$repo_dir")" != "$name" ]; then
            echo "not in $name, exiting"
            exit 1
          fi
          case $(uname) in
            Darwin) suffix=dylib ;;
            *)      suffix=so    ;;
          esac
          source=$repo_dir/target/release/maturin/lib$name.$suffix
          target=$repo_dir/python/$name/_internal.so

          if [ -e "$target" ]; then
            for other in $(find src -name '*rs'); do
              if [ "$target" -ot "$other" ]; then
                rm -f "$target"
                break
              fi
            done
          fi

          if [ ! -e "$source" -o ! -e "$target" ]; then
            maturin build --release
          fi
          if [ ! -L "$target" -o "$(realpath "$source")" != "$(realpath "$target")" ]; then
            rm -f "$target"
            ln -s "$source" "$target"
          fi
        '';

        cargoArtifacts = craneLib.buildDepsOnly commonArgs;
        trusty = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
        });

        datasets = {
          diamonds = pkgs.fetchurl {
            url = "https://raw.githubusercontent.com/tidyverse/ggplot2/master/data-raw/diamonds.csv";
            sha256 = "sha256-lXRzCwOrokHYmcSpdRHFBhsZNY+riVEHdPtsJBaDRcQ=";
          };

          airline = pkgs.fetchurl {
            url = "https://raw.githubusercontent.com/varundixit4/Airline-Passenger-Satisfaction-Report/refs/heads/main/airline_satisfaction.csv";
            sha256 = "sha256-oV+rbTamEj3tsDXhvBGzHye1R2cc6NJ3YudNllJ8Nk8=";
          };
        };

        datafusion-udf = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
          cargoExtraArgs = "--package examples --bin datafusion_udf";
          doCheck = false;
        });

        datafusion-udf-wrapper = pkgs.writeShellScriptBin "datafusion-udf" ''
          exec ${datafusion-udf}/bin/datafusion_udf "$@"
        '';

        dataFiles = pkgs.runCommand "trusty-data-files" { } ''
          mkdir -p $out/data
          ln -s ${datasets.diamonds} $out/data/diamonds.csv
          ln -s ${datasets.airline} $out/data/airline_satisfaction.csv
        '';

        prepare-benchmarks = pkgs.writeShellScriptBin "prepare-benchmarks" ''
          mkdir -p data
          echo "Copying data files from Nix store..."
          cp -f ${dataFiles}/data/* data/
          # must use -editable else we rebuild every time flake.nix changes
          ${venv-editable-312}/bin/python -m trusty.generate_examples --data_dir data --base_dir data --generation_type benchmark

        '';
        clippy-hook = pkgs.writeShellScriptBin "clippy-hook" ''
          exec ${rustToolchain}/bin/cargo clippy --all-targets --all-features -- -D warnings
        '';

        pre-commit-check = pre-commit-hooks.lib.${system}.run {
          src = ./.;
          hooks = {
            nixpkgs-fmt.enable = true;
            rustfmt.enable = true;
            ruff = {
              enable = true;
              files = "\\.py$";
              excludes = [ ];
            };
            clippy = {
              enable = true;
              entry = toString clippy-hook;
            };
          };
          tools = {
            inherit (pkgs) ruff;
            rustfmt = rustToolchain;
            clippy = rustToolchain;
          };
        };
      in
      {
        apps = rec {
          ipython = {
            type = "app";
            program = "${venv-312}/bin/ipython";
          };
          default = ipython;
        };
        lib = {
          inherit venv-312 venv-editable-312;
          inherit pythonSet pythonSet-editable;
        };
        packages = {
          inherit trusty;
          data = dataFiles;
          datafusion-udf-example = datafusion-udf-wrapper;
          default = datafusion-udf-wrapper;
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

            buildPhase = ''
              ${pre-commit-check.buildCommand}
            '';

            installPhase = ''
              touch $out
            '';

            RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
          };
        };
        devShells = {
          default = self.devShells.${system}.venv-editable-312;
          venv-312 = pkgs.mkShell {
            packages = [
              venv-312
              pkgs.uv
              rustToolchain
              prepare-benchmarks
            ];
            shellHook = ''
              unset PYTHONPATH
              export UV_PYTHON_DOWNLOADS=never
            '';
          };
          venv-editable-312 = pkgs.mkShell {
            packages = [
              venv-editable-312
              pkgs.uv
              rustToolchain
              prepare-benchmarks
            ];
            shellHook = ''
              unset PYTHONPATH
              export UV_NO_SYNC=1
              export UV_PYTHON_DOWNLOADS=never
              export REPO_ROOT=$(git rev-parse --show-toplevel)
            '' + maybeMaturinBuildHook;
          };
          impure = pkgs.mkShell {
            packages = [
              python
              pkgs.uv
              rustToolchain
            ];
            shellHook = ''
              unset PYTHONPATH
              export UV_PYTHON_DOWNLOADS=never
            '';
          };
          p2n = pkgs.mkShell {
            inputsFrom = [ trusty ];
            buildInputs = [
              rustToolchain
              pkgs.maturin
              # add pre-commit dependencies
              pkgs.ruff
              pkgs.rustfmt
              pkgs.nixpkgs-fmt
              prepare-benchmarks
            ];
            shellHook = ''
              ${pre-commit-check.shellHook}
              echo "Run 'build-maturin' to rebuild and install the package"
            '';
          };
        };
      });
}
