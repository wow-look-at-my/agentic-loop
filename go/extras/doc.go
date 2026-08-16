// Package extras holds the policies a Provider can be given but does not need
// to speak a dialect: retrying a transient failure, and staying under an
// upstream's per-minute request cap.
//
// They live outside core because core translates and does not decide. Both are
// decorators over a [commonai.Provider], so a caller assembles what it wants
// and nothing is imposed on a caller that wants neither.
package extras
