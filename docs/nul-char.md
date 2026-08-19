# `&#0;`

XML 1.1's `Char` production starts at `#x1`, so U+0000 is not a character XML
admits at all -- not literally, and not by reference. This project deviates at
exactly one point: **a character reference resolving to U+0000 is written and
accepted.** A literal NUL byte is still rejected, and so is a lone surrogate,
which is not a character in any encoding.

## Why the deviation is safe

`&#0;` is four ASCII bytes: `26 23 30 3B`. A document containing one contains
no NUL byte. Every argument for the exclusion is an argument about NUL in a
byte stream -- C strings ending early, a length that stops being the length --
and none of them reach a document that never carries the byte. What reads such
a document sees an escape sequence, decides what its own value type does with
U+0000, and is no worse off than it was for `&#1;`, which the spec requires to
be written this way for the same class of reason.

## Why we need it

A tool result is bytes a program produced, and programs produce NUL. The
alternatives were all worse:

- drop it -- a wrong answer that looks like a right one;
- substitute U+FFFD -- the same, plus a fabricated character;
- base64 the whole part -- unreadable content, for one byte;
- fail the call -- a model call refused over a byte the model handled fine.

Losslessness is the format's whole claim (`docs/format.md`). A value that
cannot survive a round trip is not one the format may quietly decline to
carry.

## Where it is implemented

- `xml-validator`: `IsCharRefValue` in `validator/chars.go` -- `IsChar` plus
  U+0000, used only on the character-reference path, so literal bytes still go
  through `IsChar`.
- Here: `mustReference` in `go/core/xmlwrite.go` -- NUL and the XML 1.1
  restricted set are written as references, everything else literally.

Our own writer and our own parser are why this works at all. An off-the-shelf
one on either end would reject the document, which is the cost of the
deviation and the reason it is confined to one code point.
