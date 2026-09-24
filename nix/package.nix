# Build one-grep with nixpkgs rustPlatform.
# OSS-safe: no private hostnames, product names, or user paths.
{
  lib,
  stdenv,
  rustPlatform,
  pkg-config,
  openssl,
  onnxruntime,
  makeWrapper,
}:

let
  ortDylib =
    if stdenv.hostPlatform.isDarwin then "libonnxruntime.dylib" else "libonnxruntime.so";
  libPathVar = if stdenv.hostPlatform.isDarwin then "DYLD_LIBRARY_PATH" else "LD_LIBRARY_PATH";
in

rustPlatform.buildRustPackage {
  pname = "one-grep";
  version = "0.2.0";

  src = lib.cleanSourceWith {
    src = ../.;
    filter =
      path: _type:
      let
        name = baseNameOf path;
      in
      name != "target"
      && name != ".git"
      && name != ".one-grep"
      && name != ".DS_Store"
      && name != "build-agent-onegrep.log"
      && !(lib.hasSuffix ".md" name && lib.hasPrefix "RECEIPT-" name)
      && !(lib.hasSuffix ".md" name && lib.hasPrefix "PACKET-" name);
  };

  cargoLock.lockFile = ../Cargo.lock;

  nativeBuildInputs = [
    pkg-config
    rustPlatform.bindgenHook
    makeWrapper
  ];

  buildInputs = [
    openssl
    onnxruntime
  ];

  # Prefer nixpkgs onnxruntime over ort-sys network download (sandbox-safe).
  # nixpkgs ships a shared library only; without ORT_PREFER_DYNAMIC_LINK,
  # ort-sys looks for static archives and fails with "could not link".
  # Darwin Security/SystemConfiguration come from the stdenv apple-sdk
  # (do not reference legacy darwin.apple_sdk.frameworks stubs).
  ORT_STRATEGY = "system";
  ORT_LIB_LOCATION = "${onnxruntime}/lib";
  ORT_PREFER_DYNAMIC_LINK = "1";

  # Package check deferred; use `cargo test` / host validation instead.
  doCheck = false;

  postInstall = ''
    wrapProgram $out/bin/one-grep \
      --prefix ${libPathVar} : ${lib.makeLibraryPath [ onnxruntime ]} \
      --set-default ORT_DYLIB_PATH ${onnxruntime}/lib/${ortDylib}
    ln -s one-grep $out/bin/open-grep
  '';

  meta = with lib; {
    description = "Local-first hybrid workspace search (BM25 + ONNX embeddings + MCP)";
    license = licenses.asl20;
    mainProgram = "one-grep";
    platforms = platforms.unix;
  };
}
