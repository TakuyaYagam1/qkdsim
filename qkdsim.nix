{
  pkgs ? import <nixpkgs> {
    config.allowUnfreePredicate = pkg: (pkg.pname or "") == "qkdsim";
  },
}:

pkgs.stdenvNoCC.mkDerivation {
  pname = "qkdsim";
  version = "1.1.10.754";

  src =
    ./.
    + "/HighTech2026_without_ova/New_QKDSim/1.1_(10.754)_R/linux/QKDSim-1.1.10.754-debian_bookworm.deb";

  nativeBuildInputs = with pkgs; [
    autoPatchelfHook
    binutils
    makeWrapper
  ];

  buildInputs = with pkgs; [
    stdenv.cc.cc.lib
    libGL
    libice
    libsm
    libx11
    libxcomposite
    libxcb
    xcbutilimage
    xcbutilkeysyms
    xcbutilrenderutil
    xcbutilwm
    libdrm
    libinput
    mtdev
    udev
    wayland
  ];

  runtimeDependencies = map pkgs.lib.getLib (
    with pkgs;
    [
      dbus
      fontconfig
      freetype
      libusb1
      libxkbcommon
      udev
    ]
  );

  unpackPhase = ''
    runHook preUnpack
    ar p "$src" data.tar.gz | tar -xz
    runHook postUnpack
  '';

  dontConfigure = true;
  dontBuild = true;
  dontStrip = true;

  installPhase = ''
    runHook preInstall

    mkdir -p "$out/bin" "$out/libexec" "$out/share"
    cp -a opt/infotecs/SimQKD-VA "$out/libexec/qkdsim"
    cp -a usr/share/icons usr/share/pixmaps usr/share/doc "$out/share/"
    app="$out/libexec/qkdsim"

    # These optional backends require Qt modules absent from the Debian package.
    rm "$app/bin/archdatadir/plugins/platforms/libqeglfs.so" \
       "$app/bin/archdatadir/plugins/platforms/libqvnc.so"

    # The bundled Fontconfig cannot parse the current system configuration.
    cat > "$app/fonts.conf" <<EOF
    <?xml version="1.0"?>
    <fontconfig>
      <dir>${pkgs.dejavu_fonts}/share/fonts/truetype</dir>
      <dir>/run/current-system/sw/share/X11/fonts</dir>
      <dir prefix="xdg">fonts</dir>
      <cachedir prefix="xdg">qkdsim/fontconfig</cachedir>
      <alias><family>sans-serif</family><prefer><family>DejaVu Sans</family></prefer></alias>
      <alias><family>serif</family><prefer><family>DejaVu Serif</family></prefer></alias>
      <alias><family>monospace</family><prefer><family>DejaVu Sans Mono</family></prefer></alias>
    </fontconfig>
    EOF

    # Use XWayland: the bundled Wayland/EGL backend crashes on this desktop.
    makeWrapper "$app/bin/SimQKD-VA" "$out/bin/qkdsim" \
      --prefix LD_LIBRARY_PATH : "$app/lib" \
      --set QT_PLUGIN_PATH "$app/bin/archdatadir/plugins" \
      --set QT_QPA_PLATFORM_PLUGIN_PATH "$app/bin/archdatadir/plugins/platforms" \
      --set QT_QPA_PLATFORM xcb \
      --set-default FONTCONFIG_FILE "$app/fonts.conf" \
      --set-default XKB_CONFIG_ROOT "${pkgs.xkeyboard_config}/share/X11/xkb" \
      --set-default XLOCALEDIR "${pkgs.libx11}/share/X11/locale" \
      --unset QT_QPA_PLATFORMTHEME \
      --unset QT_STYLE_OVERRIDE

    install -Dm644 "usr/share/applications/ViPNet QKDSim.desktop" \
      "$out/share/applications/qkdsim.desktop"
    substituteInPlace "$out/share/applications/qkdsim.desktop" \
      --replace-fail 'Exec=/opt/infotecs/SimQKD-VA/bin/simqkd_without_brltty.sh' \
        "Exec=$out/bin/qkdsim"

    runHook postInstall
  '';

  meta = {
    description = "ViPNet QKDSim quantum key distribution simulator";
    license = pkgs.lib.licenses.unfree;
    platforms = [ "x86_64-linux" ];
    mainProgram = "qkdsim";
  };
}
