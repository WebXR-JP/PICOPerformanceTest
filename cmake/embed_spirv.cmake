# SPIR-V バイナリを C++ の uint32_t 配列ヘッダーに変換するスクリプト
# 使い方:
#   cmake -DINPUT=xxx.spv -DOUTPUT=xxx_spv.h -DVAR_NAME=xxx_spv -P embed_spirv.cmake

file(READ "${INPUT}" raw_content HEX)

# 2文字ずつ区切ってバイトリストを作成
string(REGEX MATCHALL ".." bytes "${raw_content}")
list(LENGTH bytes byte_count)

# 4バイトずつ uint32_t にまとめる
set(words "")
set(i 0)
foreach(b ${bytes})
    math(EXPR mod "${i} % 4")
    if(mod EQUAL 0)
        set(word "0x")
    endif()
    # リトルエンディアン: バイトを逆順に並べる
    set(word_bytes_${mod} "${b}")
    if(mod EQUAL 3)
        set(w "0x${word_bytes_3}${word_bytes_2}${word_bytes_1}${word_bytes_0}")
        list(APPEND words "${w}")
    endif()
    math(EXPR i "${i} + 1")
endforeach()

# 端数バイト処理（4の倍数でない場合）
math(EXPR remainder "${byte_count} % 4")
if(NOT remainder EQUAL 0)
    if(remainder EQUAL 1)
        set(w "0x000000${word_bytes_0}")
    elseif(remainder EQUAL 2)
        set(w "0x0000${word_bytes_1}${word_bytes_0}")
    elseif(remainder EQUAL 3)
        set(w "0x00${word_bytes_2}${word_bytes_1}${word_bytes_0}")
    endif()
    list(APPEND words "${w}")
endif()

list(JOIN words ", " words_str)
math(EXPR word_count "(${byte_count} + 3) / 4")

file(WRITE "${OUTPUT}"
"#pragma once
#include <cstdint>
// Auto-generated from SPIR-V binary. Do not edit.
// Source: ${INPUT}
static const uint32_t ${VAR_NAME}[] = {
    ${words_str}
};
static const uint32_t ${VAR_NAME}_size = ${byte_count};
static const uint32_t ${VAR_NAME}_word_count = ${word_count};
")
