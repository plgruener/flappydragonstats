#!/usr/bin/env fish

# clean the dumped JSON files from discord:
# - CRLF -> LF line endings
# - fix missing newline at EOF
# - remove stray '\r' in strings
# - remove stray whitespace in strings
# - correct `type_pool` to "All" for Chrysalis, Pterodactyl, Tetric egg
# - change key `egg_subtype` -> `category` (more accurate)

set input $argv[1]

dos2unix $input

if test (tail -c1 $input | xxd -p) != '0a'
  echo '' >> $input
end

sed -i -e 's/\\\\r//g' $input
sed -i -e 's/ "$/"/' $input
sed -i -e 's/ ",$/",/' $input


if test (basename -s .json $input | string split -f1 '_') = 'eggs'
  jq --indent 4 'map(if .name.en | IN("Chrysalis Egg", "Pterodactyl Egg", "Tetric egg") then .type_pool = "All" else . end )' $input | sponge $input
end

sed -i -e 's/egg_subtype/category/g' $input
