## What's new in LuaRocks 3.5.0

This is a small release:

* Added support for MSYS2 and Mingw-w64
* Reverted the change in MSVC environment variable set up script
* Fixes a bug where `--verbose` raised an exception with a nil argument
* Added proper error messages when lua.h is invalid


## What's new in LuaRocks 3.4.0

### Features

* `luarocks make` now supports `--only-deps`
* `luarocks make` new flag: `--no-install`, which only performs
  the compilation step
* `--deps-only` is now an alias for `--only-deps` (useful in case
  you always kept getting it wrong, like me!)
* `luarocks build` and `luarocks make` now support using
  `--pin` and `--only-deps` at the same time, to produce a lock
  file of dependencies in use without installing the main package.
* `luarocks show` can now accept a substring of the rock's name,
  like `list`.
* `luarocks config`: when running without system-wide permissions,
  try storing the config locally by default.
  Also, if setting both lua_dir and --lua-version explicitly,
  auto-switch the default Lua version.
* `luarocks` with no arguments now prints more info about the
  location of the Lua interpreter which is being used
* `luarocks new_version` now keeps the old URL if the MD5 doesn't
  change.
* `DEPS_DIR` is now accepted as a generic variable for dependency
  directories (e.g. `luarocks install foo DEPS_DIR=/usr/local`)
* Handle quoting of arguments at the application level, for
  improved Windows support
* All-in-one binary bundles `dkjson`, so it runs `luarocks upload`
  without requiring any additional dependencies.
* Tweaks for Terra compatibility

### Fixes

* win32: generate proper temp filename
* No longer assume that Lua 5.3 is built with compat libraries and
  bundles `bit32`
* `luarocks show`: do not crash when rockspec description is empty
* When detecting the location of `lua.h`, check that its version
  matches the version of Lua being used
* Fail gracefully when a third-party tool (wget, etc.) is missing
* Fix logic for disabling mirrors that return network errors
* Fix detection of Lua path based on arg variable
* Fix regression on dependency matching of luarocks.loader


## What's new in LuaRocks 3.3.1

This is a bugfix release:

* Fix downgrades of rocks containing directories: stop it
  from creating spurious 0-byte files where directories have been 
* Fix error message when attempting to copy a file that is missing
* Detect OpenBSD-specific dependency paths

## What's new in LuaRocks 3.3.0

### Features

* **Dependency pinning**
  * Adds a new flag called `--pin` which creates a `luarocks.lock`
    when building a rock with `luarocks build` or `luarocks make`.
    This lock file contains the exact version numbers of every
    direct or indirect dependency of the rock (in other words,
    it is the transitive closure of the dependencies.)
    For `make`, the `luarocks.lock` file is created in the current
    directory.
    The lock file is also installed as part of the rock in
    its metadata directory alongside its rockspec.
    When using `--pin`, if a lock file already exists, it is
    ignored and overwritten.
  * When building a rock with `luarocks make`, if there is a
    `luarocks.lock` file in the current directory, the exact
    versions specified there will be used for resolving dependencies.
  * When building a rock with `luarocks build`, if there is a
    `luarocks.lock` file in root of its sources, the exact
    versions specified there will be used for resolving dependencies.
  * When installing a `.rock` file with `luarocks install`, if the
    rock contains a `luarocks.lock` file (i.e., if its dependencies
    were pinned with `--pin` when the rock was built), the exact
    versions specified there will be used for resolving dependencies.
* Improved VM type detection to support moonjit
* git: Support for shallow recommendations
* Initial support for Windows on ARM
* Support for building 64-bit Windows all-in-one binary
* More filesystem debugging output when using `--verbose` (now it
  reports operations even when using LuaFileSystem-backed implementations)
* `--no-manifest` flag for creating a package without updating the
  manifest files
* `--no-doc` flag is now supported by `luarocks make`

### Performance improvements

* Speed up dependency checks
* Speed up installation and deletion when deploying files
* build: do not download sources when when building with `--only-deps`
* New flag `--check-lua-versions`: when a rock name is not found, only
  checks for availability in other Lua versions if this flag is given

### Fixes

* safer rollback on installation failure
* config: fix `--unset` flag
* Fix command name invocations with dashes (e.g. `luarocks-admin make-manifest`)
* Fix fallback to PATH search when Lua interpreter is not configured
* Windows: support usernames with spaces
* Windows: fix generation of temporary filenames (#1058)
* Windows: force `.lib` over `.dll` extension when resolving `LUALIB`

## What's new in LuaRocks 3.2.1

* fix installation of LuaRocks via rockspec (`make bootstrap` and
`luarocks install`): correct a problem in the initialization of the
luarocks.fs module and its interaction with the cfg module.
* fix luarocks build --pack-binary-rock --no-doc
* fix luarocks build --branch
* luarocks init: fix Lua wrapper for interactive mode
* fix compatibility issues with command add-ons loaded via
luarocks.cmd.external modules
* correct override of config values via CLI flags

## What's new in LuaRocks 3.2.0

LuaRocks 3.2.0 now uses argument parsing based on argparse
instead of a homegrown parser. This was implemented by Paul
Ouellette as his Google Summer of Code project, mentored by
Daurnimator.

Release highlights:

* Bugfix: luarocks path does not change the order of pre-existing path
items when prepending or appending to path variables
* Bugfix: fix directory detection on the Mac
* When building with --force-config, LuaRocks now never uses the
"project" directory, but only the forced configuration
* Lua libdir is now only checked for commands/platforms that really
need to link Lua explicitly
* LuaJIT is now detected dynamically
* RaptorJIT is now detected as a LuaJIT variant
* Improvements in Lua autodetection at runtime
* luarocks new_version: new option --dir
* luarocks which: report modules found via package.path and
package.cpath as well
* install.bat: Improved detection for Visual Studio 2017 and higher
* Bundled LuaSec in all-in-one binary bumped to version 0.8.1

## What's new in LuaRocks 3.1.3

This is another bugfix release, that incldes a couple of fixes,
including better Lua detection, and fixes specific to MacOS and
FreeBSD.

## What's new in LuaRocks 3.1.2

This is again a small fix release.

## What's new in LuaRocks 3.1.1

This is a hotfix release fixing an issue that affected initialization
in some scenarios.

## What's new in LuaRocks 3.1.0

### More powerful `luarocks config`

The `luarocks config` command used to only list the current
configuration. It is now able to query and also _set_ individual
values, like `git config`. You can now do things such as:

   luarocks config variables.OPENSSL_DIR /usr/local/openssl
   luarocks config lua_dir /usr/local
   luarocks config lua_version 5.3

and it will rewrite your luarocks configuration to store that value
for later reuse. Note that setting `lua_version` will make that Lua
version the default for `luarocks` invocations (you can always
override on a per-call basis with `--lua-version`.

You can specify the scope where you will apply the configuration
change: system-wide, to the user's home config (with --local), or
specifically to a project, if you run the command from within a
project directory initialized with `luarocks init`.

### New `--global` flag

Some users prefer that LuaRocks default to system-wide installations,
some users prefer to install everything to their home directory. The
`local_by_default` configuration file controls this preference: when
it is off, the `--local` file triggers user-specific. Before 3.1.0
there was no convenient way to trigger system-wide installations when
`local_by_default` was set to true. LuaRocks 3.1.0 adds a `--global`
flag to this purpose. To enable local-by-default, you can now do:

   luarocks config local_by_default true

### `luarocks make` can deal with patches

A rockspec can include embedded patch files, which are applied when a
source rock is built. Now, when you run `luarocks make` on a source
tree unpacked with `luarocks unpack`, the patches will be applied as
well (and a hidden lockfile is created to avoid the patches to be
re-applied incorrectly).

### Smarter defaults when working with projects

When working on a project initialized with `luarocks init`, the
presence of a ./.luarocks/config-5.x.lua file will be enough to detect
the project-based workflow and have `luarocks` default to that 5.x
version. That means the `./luarocks` wrapper becomes less necessary;
the `luarocks` from your $PATH will deal with the project just fine,
git-style.

### And more!

There are also other improvements. LuaRocks uses the manifest cache a
bit more aggressively, resulting in increased performance. Also, it no
longer complains with a warning message if the home cache cannot be
created (it just uses a temporary dir instead). And of course, the
release includes multiple bugfixes.

## What's new in LuaRocks 3.0.4

* Fork-free platform detection at startup
* Improved detection of the default rockspec in commands such as `luarocks test`
* Various minor bugfixes

## What's new in LuaRocks 3.0.3

LuaRocks 3.0.3 is a minor bugfix release, fixing a regression in
luarocks.loader introduced in 3.0.2.

## What's new in LuaRocks 3.0.2

* Improvements in luarocks init, new --reset flag
* write_rockspec: --lua-version renamed to --lua-versions
* Improved behavior in module autodetection
* Bugfixes in luarocks show
* Fix upgrade/downgrade when a single rock has clashing module
filenames (should fix the issue when downgrading luasec)
* Fix for autodetected external dependencies with non-alphabetic
characters (should fix the libstdc++ issue when installing xml)


## What's new in LuaRocks 3.0.1

* Numerous bugfixes including:
   * Handle missing global `arg`
   * Fix umask behavior
   * Do not overwrite paths in format 5.x.y when cleaning up path
variables (#868)
   * Do not detect files under lua_modules as part of your sources
when running `luarocks write_rockspec`
   * Windows: do not hardcode MINGW in the all-in-one binary: instead
it properly detects when running from a Visual Studio Developer
Console and uses that compiler instead
   * configure: --sysconfdir was fixed to its correct meaning: it now
defaults to /etc and not /etc/luarocks (`/luarocks` is appended to the
value of sysconfdir)
   * configure: fixed --force-config
* Store Lua location in config file, so that a user can run `luarocks
init --lua-dir=/my/lua/location` and have that location remain active
for that project
* Various improvements to the Unix makefile, including $(DESTDIR)
support and an uninstall rule
* Autodetect FreeBSD-style include paths (/usr/include/lua5x/)

## What's new in LuaRocks 3.0.0

- [New rockspec format](#new-rockspec-format)
- [New commands](#new-commands), including [luarocks init](https://github.com/luarocks/luarocks/wiki/Project:-LuaRocks-per-project-workflow) for per-project workflows
- [New flags](#new-flags), including `--lua-dir` and `--lua-version` for using multiple Lua installs with a single LuaRocks
- [New build system](#new-build-system)
- [General improvements](#general-improvements), including [namespaces](https://github.com/luarocks/luarocks/wiki/Namespaces)
- [User-visible changes](#user-visible-changes), including some **breaking changes**
- [Internal changes](#internal-changes)

### New rockspec format

**New rockspec format:** if you add `rockspec_format = "3.0"` to your rockspec,
you can use a number of new features. Note that these rockspecs will only work
with LuaRocks 3.0 and above, but older versions will detect that directive and
fail gracefully, giving the user a message telling them to upgrade. Rockspecs
without the `rockspec_format` directive are interpreted as having format 1.0
(the same format from LuaRocks series 1.x and 2.x) and are still supported.

The following features are only enabled if `rockspec_format = "3.0"` is set in
the rockspec:

* Build type `builtin` is the default if `build.type` is not specified.
* The `builtin` type auto-detects modules using the same heuristics as
  `write_rockspec` (for example, if you have a `src` directory). With
  auto-detection of the build type and modules, many rockspecs don't
  even need an explicit `build` table anymore.
* New table `build_dependencies`: dependencies used only for running
  `luarocks build` but not when installing binary rocks.
* New table `test_dependencies`: dependencies used only for running `luarocks test`
* New table `test`: settings for configuring the behavior of `luarocks test`.
  Supports a `test.type` field so that the test backend can be specified.
  Currently supported test backends are:
  * `"busted"`, for running [Busted](https://olivinelabs.com/busted)
  * `"command"`, for running a plain command.
  * Custom backends can be loaded via `test_dependencies`
* New field `build.macosx_deployment_target = "10.9"` is supported in Mac platforms,
  and adjusts `$(CC)` and `$(LD)` variables to export the corresponding
  environment variable.
* LuaJIT can be detected in dependencies and uses version reported by the
  running interpreter: e.g. `"luajit >= 2.1"`.
* Auto-detection of `source.dir` is improved: when the tarball contains
  only one directory at the root, assume that is where the sources are.
* New `description` fields:
  * `labels`, an array of strings;
  * `issues_url`, URL to the project's bug tracker.
* `cmake` build type now supports `build.build_pass` and `build_install_pass`
  to disable `make` passes.
* `git` fetch type fetches submodules by default.
* Patches added in `patches` can create and delete files, following standard
  patch rules.

### New commands

* **New command:** `luarocks init`. This command performs the setup for using
  LuaRocks in a "project directory":
  * it creates a `lua_modules` directory in the current directory for
    storing rocks
  * it creates a `.luarocks/config-5.x.lua` local configuration file 
  * it creates `lua` and `luarocks` wrapper scripts in the current
    directory that are configured to use `lua_modules` and
    `.luarocks/config-5.x.lua`
  * if there are no rockspecs in the current directory, it creates one
    based on the directory name and contents.
* **New command:** `luarocks test`. It runs a rock's test suite, as specified
  in the new `test` section of the rockspec file. It also does some
  autodetection, so it already works with many existing rocks as well.
* **New command:** `luarocks which`. Given the name of an installed, it tells
  you which rock it is a part of. For example, `luarocks which lfs`
  will tell you it is a part of `luafilesystem` (and give the full
  path name to the module). In this sense, `luarocks which` is the
  dual command to `luarocks show`.

### New flags

* **New flags** `--lua-dir` and `--lua-version` which can be used with
  all commands. This allows you to specify a Lua version and installation
  prefix at runtime, so a single LuaRocks installation can be used
  to manage packages for any Lua version. It is no longer necessary to
  install separate copies of LuaRocks to manage packages for Lua 5.x
  and 5.y.
* **New flags** added to `luarocks show`: `--porcelain`, giving a stable
  script-friendly output (named after the Git `--porcelain` flag that
  serves the same purpose) and `--rock-license`.
* **New flag** `--temp-key` for `luarocks upload`, allowing you to easily
  upload rocks into an alternate account without disrupting the
  stored configuration of your main account.
* **New flag** `--dev`, for enabling development-branch sub-repositories.
  This adds support for easily requesting `dev` modules from LuaRocks.org, as in:
  `luarocks install --dev luafilesystem`. The list of URLs configured
  in `rocks_servers` is prepended with a list containing "/dev" in their paths.
* `luarocks config`, when called with no arguments, now displays your
  entire active configuration, using the same Lua syntax as the configuration
  file. It is sensitive to the flags given to it (`--tree`, `--lua-dir`, etc.)
  so it presents the resulting configuration produced by loading the
  currently-active configuration files and the given flags.

### New build system

**New build system**: the `configure` and `Makefile` scripts were completely
overhauled, making use of LuaRocks 3 features to greatly simplify them:

* Much of the detection and configuration work they performed were moved
  to runtime, to make LuaRocks more dynamic and resilient to environment
  changes
* The system-package-manager-friendly mode is still available, as the
  default target (`make`, formerly `make build`).
* The LuaRocks-as-a-rock mode (`make bootstrap`) is also still available,
  and was greatly simplified: it no longer uses custom Makefiles:
  LuaRocks installs itself using `luarocks make`, and its own rockspec
  uses the `builtin` build mode.
* A new build mode: `make binary` compiles all of LuaRocks into a single
  executable, bundling various Lua modules to make it self-sufficient,
  such as LuaFileSystem, LuaSocket and LuaSec.
  * For version 3.0, this will remain as an option, as we evaluate
    its suitability moving forward to become the default mode of
    distribution.
  * The goal is to eventually use this mode to produce the Windows
    version of LuaRocks. We currently include an experimental
    `make windows-binary` target which builds a Windows version
    using the MinGW-w64 cross-compiler on Linux.

### General improvements

* **New feature:** [namespaces](https://github.com/luarocks/luarocks/wiki/Namespaces):
  you can use `luarocks install user/package` to install a package from a
  specific user of the repository.
* Improved defaults for finding external libraries on Linux and Windows.
* Detection of the Lua library and header directories is now done at runtime.
  This uses the same machinery that LuaRocks employs for `external_dependencies`
  in general (with some added logic to cope with the unfortunate
  rampant inconsistency in naming of Lua libraries and header paths
  due to lack of upstream standardization).
* `luarocks-admin add` now works with `file://` repositories
* some UI improvements in `luarocks list` and `luarocks search`.
* Preliminary support for the upcoming Lua 5.4: LuaRocks is written in
  the common dialect supporting Lua 5.1-5.3 and LuaJIT, but since a
  single installation can manage packages for any Lua version now,
  it can already manage packages for Lua 5.4 even though that's not
  out yet.

### User-visible changes

* **Breaking change:** The support for deprecated unversioned paths
  (e.g. `/usr/local/lib/luarocks/rocks/` and `/etc/luarocks/config.lua`)
  was removed, LuaRocks will now only create and use paths versioned
  to the specific Lua version in use
  (e.g. `/usr/local/lib/luarocks/rocks-5.3/` and `/etc/luarocks/config-5.3.lua`).
* **Breaking changes:** `luarocks path` now exports versioned variables
  `LUA_PATH_5_x` and `LUA_CPATH_5_x` instead of `LUA_PATH` and `LUA_CPATH`
  when those are in use in your system.
* Package paths are sanitized to only reference the current Lua version.
  For example, if you have `/some/dir/lua/5.1/` in your `$LUA_PATH` and
  you are running Lua 5.2, `luarocks.loader` and the `luarocks` command-line
  tool will convert it to `/some/dir/lua/5.2/`.
* LuaRocks now uses `dev` instead of `scm` as the favored version identifier
  to describe development versions of a rock, aligning it with the terminology
  used in https://luarocks.org. It still understands `scm` as a
  compatibility fallback.
* LuaRocks no longer conflates modules `foo` and `foo.init` as being the
  same in its internal manifest. Instead, the `luarocks.loader` module
  is adapted to handle the `.init` case.
* Wrappers installed using `--tree` now prepend the tree's prefix to their
  package paths.
* `luarocks-admin` commands no longer creates an `index.html` file in the
  repository by default (it does update it if it already exists)

### Internal changes

* Major improvements in the test suite done by @georgeroman as part of the ongoing
  Google Summer of Code 2018 program. The coverage improvements and test suite
  speed-ups have been essential in getting the sprint towards LuaRocks 3.0 more
  efficient and reliable!
* Modules needed by `luarocks.loader` were moved below the `luarocks.core` namespace.
  Modules in `luarocks.core` only depend on other `luarocks.core` modules.
  (Notably, `luarocks.core` does not use `luarocks.fs`.)
* Modules representing `luarocks` commands were moved into the `luarocks.cmd` namespace,
  and `luarocks.command_line` was renamed to `luarocks.cmd`. Eventually, all CLI-related
  code will live under `luarocks.cmd`, as we move towards a clean CLI-API separation,
  in preparation for a stable public API.
* Likewise, modules representing `luarocks-admin` commands were moved into the
  `luarocks.admin.cmd` namespace.
* New internal objects for representing interaction with the repostories:
  `luarocks.queries` and `luarocks.results`
* Type checking rules of file formats were moved into the `luarocks.type` namespace.
